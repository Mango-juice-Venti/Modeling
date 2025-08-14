# -*- coding: utf-8 -*-
"""
리조트 식음업장 — XGBoost 듀얼 회귀 앙상블(Tweedie + log1p) + 시간누수 방지 타깃인코딩
+ 업장가중 SMAPE(공식식) 검증 + α블렌딩 + (item×dow 포함) 다단 캘리브레이션
+ 멀티시드 배깅 + 희소아이템 블렌딩 + 아이템별 p95 상한 + 7일 재귀예측 + 퍼지매칭 제출

이전 요청 반영:
- POS_SAMPLE_BOOST = 4.0
- CAL_CLIP = (0.85, 1.15)
- SPARSE_NZ_THRESHOLD = 0.10, STALE_DAYS_THRESHOLD = 21
- alphas = np.linspace(0.1, 0.9, 17)

신규 강화:
- SEEDS = [42, 77, 2023] 배깅(역SMAPE가중)
- 요일 단위 캘리브레이션(item×dow, shop×dow)
- item별 최근56일 p95 * 1.10 상한 클리핑
"""
import os, glob, re, unicodedata, difflib
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import xgboost as xgb

# ====================== Config ======================
BASE_DIR   = Path(r"C:\Users\minseo\lg")
TRAIN_FILE = BASE_DIR / "train" / "re_train.csv"
TEST_GLOB  = str(BASE_DIR / "test" / "TEST_*processed.csv")
SAMPLE_SUB = BASE_DIR / "sample_submission.csv"
OUT_FILE   = BASE_DIR / "submission_6.csv"

RANDOM_STATE = 42
VALID_LAST_DAYS = 28
SEEDS = [42, 77, 2023]             # 멀티시드 배깅

# 업장 가중치(평가식의 w_s 근사)
SHOP_WEIGHTS = {"담하": 2.0, "미라시아": 2.0}  # 그 외 1.0

# Tweedie·학습 가중
POS_SAMPLE_BOOST = 4.0

# 캘리브레이션(검증 합계 비율 보정)
CAL_CLIP = (0.85, 1.15)

# 희소아이템 블렌딩
SPARSE_NZ_THRESHOLD = 0.10
STALE_DAYS_THRESHOLD = 21
BLEND_MODEL = 0.60
BLEND_ANCHOR_RATIO = 0.40

# 후처리(완만)
SOFT_ZERO_CUT = 0.0
WEEKEND_BUMP  = 1.05
HOLIDAY_BUMP  = 1.10
SPIKE_BUMP    = 1.06
SPIKE_RATIO   = 1.50

# 아이템 상한(과대예측 컷)
CAP_LOOKBACK = 56
CAP_P = 95
CAP_MULT = 1.10
ENABLE_CAP = True

np.set_printoptions(suppress=True)

# ====================== Utils ======================
def safe_read_csv(path: Path):
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try: return pd.read_csv(path, encoding=enc)
        except Exception: pass
    raise RuntimeError(f"Failed to read {path}")

def ensure_upjang(df):
    if "영업장명" not in df.columns:
        if "영업장명_메뉴명" in df.columns:
            df["영업장명"] = df["영업장명_메뉴명"].astype(str).str.split("_", n=1).str[0]
        else:
            df["영업장명"] = ""
    return df

def smape_ignore_zero(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = y_true != 0
    if m.sum() == 0: return 0.0
    yt, yp = y_true[m], y_pred[m]
    return 100.0 * np.mean(2.0 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp) + eps))

# ====================== Feature Engineering ======================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["영업일자"] = pd.to_datetime(out["영업일자"], errors="coerce")
    out = ensure_upjang(out)

    # Tweedie 제약: 음수/NaN 라벨 제거
    if "매출수량" in out.columns:
        out["매출수량"] = pd.to_numeric(out["매출수량"], errors="coerce").fillna(0.0).clip(lower=0.0)

    out = out.sort_values(["영업장명_메뉴명","영업일자"]).reset_index(drop=True)

    # Calendar
    out["dow"]   = out["영업일자"].dt.weekday
    out["week"]  = out["영업일자"].dt.isocalendar().week.astype(int)
    out["month"] = out["영업일자"].dt.month
    out["year"]  = out["영업일자"].dt.year
    out["day"]   = out["영업일자"].dt.day
    out["woy"]   = out["영업일자"].dt.isocalendar().week.astype(int)

    # Lags & Rollings
    for lag in [1,7,14,28]:
        out[f"lag_{lag}"] = out.groupby("영업장명_메뉴명")["매출수량"].shift(lag)

    for win in [7,14,28]:
        grp = out.groupby("영업장명_메뉴명")["매출수량"]
        out[f"roll_mean_{win}"] = grp.shift(1).rolling(win, min_periods=3).mean()
        out[f"roll_std_{win}"]  = grp.shift(1).rolling(win, min_periods=3).std()
        out[f"roll_max_{win}"]  = grp.shift(1).rolling(win, min_periods=3).max()
        out[f"roll_min_{win}"]  = grp.shift(1).rolling(win, min_periods=3).min()

    # Time-safe target encodings (shift(1) 후 expanding)
    out["exp_item_dow_mean"] = (
        out.groupby(["영업장명_메뉴명","dow"])["매출수량"]
           .apply(lambda s: s.shift(1).expanding(min_periods=3).mean())
           .reset_index(level=[0,1], drop=True)
    )
    out["exp_item_mean"] = (
        out.groupby("영업장명_메뉴명")["매출수량"]
           .apply(lambda s: s.shift(1).expanding(min_periods=3).mean())
           .reset_index(level=0, drop=True)
    )
    out["exp_shop_dow_mean"] = (
        out.groupby(["영업장명","dow"])["매출수량"]
           .apply(lambda s: s.shift(1).expanding(min_periods=3).mean())
           .reset_index(level=[0,1], drop=True)
    )
    out["item_dow_idx"] = out["exp_item_dow_mean"] / (out["exp_item_mean"] + 1e-6)
    out["shop_dow_idx"] = out["exp_shop_dow_mean"] / (
        out.groupby("영업장명")["exp_shop_dow_mean"].transform(lambda s: s.shift(1).expanding(min_periods=3).mean()) + 1e-6
    )

    # Trend helpers
    out["trend_7_1"]  = out["roll_mean_7"]  / (out["lag_1"] + 1e-6)
    out["trend_14_7"] = out["roll_mean_14"] / (out["roll_mean_7"] + 1e-6)
    out["delta_1_7"]  = out["lag_1"] - out["roll_mean_7"]

    out["nonzero_rate_28"] = (
        out.groupby("영업장명_메뉴명")["매출수량"]
          .transform(lambda s: s.shift(1).rolling(28, min_periods=3).apply(lambda x: (x>0).mean(), raw=True))
    )

    # 마지막 판매 이후 경과일
    def days_since_last_sale(g):
        last = None; res = []
        for d, y in zip(g["영업일자"], g["매출수량"]):
            res.append(365 if last is None else (d - last).days)
            if y > 0: last = d
        return pd.Series(res, index=g.index)

    try:
        out["days_since_last_sale"] = (
            out.groupby("영업장명_메뉴명", group_keys=False)
               .apply(days_since_last_sale, include_groups=False).astype(float)
        )
    except TypeError:
        out["days_since_last_sale"] = (
            out.groupby("영업장명_메뉴명", group_keys=False)
               .apply(days_since_last_sale).astype(float)
        )

    # Fourier
    out["dow_sin"]   = np.sin(2*np.pi*out["dow"]/7)
    out["dow_cos"]   = np.cos(2*np.pi*out["dow"]/7)
    out["month_sin"] = np.sin(2*np.pi*out["month"]/12)
    out["month_cos"] = np.cos(2*np.pi*out["month"]/12)
    out["woy_sin"]   = np.sin(2*np.pi*out["woy"]/53)
    out["woy_cos"]   = np.cos(2*np.pi*out["woy"]/53)

    # Flags
    for c in ["is_spike","is_drop","is_weekday_price","is_weekend_price","is_holiday"]:
        out[c] = out.get(c, 0)
        if c in out.columns: out[c] = out[c].fillna(0).astype(int)

    # Encodings
    out["_banquet_type_enc"] = out.get("banquet_type", -1)
    if "banquet_type" in out.columns:
        out["_banquet_type_enc"] = out["banquet_type"].astype("category").cat.codes
    out["item_id"] = out["영업장명_메뉴명"].astype("category").cat.codes
    out["업장_id"]  = out["영업장명"].astype("category").cat.codes

    # Fill NA
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if "매출수량" in num_cols: num_cols.remove("매출수량")
    out[num_cols] = out[num_cols].fillna(0.0)
    return out

def feature_columns(df: pd.DataFrame):
    base = [
        "item_id","업장_id",
        "dow","week","month","year","day","woy","woy_sin","woy_cos",
        "lag_1","lag_7","lag_14","lag_28",
        "roll_mean_7","roll_std_7","roll_max_7","roll_min_7",
        "roll_mean_14","roll_std_14","roll_max_14","roll_min_14",
        "roll_mean_28","roll_std_28","roll_max_28","roll_min_28",
        "exp_item_dow_mean","exp_item_mean","exp_shop_dow_mean",
        "item_dow_idx","shop_dow_idx",
        "trend_7_1","trend_14_7","delta_1_7",
        "nonzero_rate_28","days_since_last_sale",
        "dow_sin","dow_cos","month_sin","month_cos",
        "is_spike","is_drop","is_weekday_price","is_weekend_price","is_holiday",
        "_banquet_type_enc"
    ]
    return [c for c in base if c in df.columns]

# ====================== 업장가중 SMAPE(공식식) ======================
def weighted_smape_like_official(df_true_pred):
    def w(shop): return SHOP_WEIGHTS.get(str(shop), 1.0)
    score = 0.0
    for s, sub_s in df_true_pred.groupby("영업장명", sort=False):
        item_scores = []
        for _, sub_i in sub_s.groupby("영업장명_메뉴명", sort=False):
            m = sub_i["y_true"].values != 0
            if m.sum() == 0: continue
            item_scores.append(smape_ignore_zero(sub_i["y_true"].values, sub_i["y_pred"].values))
        if item_scores:
            score += w(s) * float(np.mean(item_scores))
    return float(score)

# ====================== Calibration ======================
def compute_calibration_from_validation(y_true, y_pred, shops, items, dows):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "영업장명": shops, "메뉴": items, "dow": dows})
    df = df[df["y_true"] > 0]

    def ratio(a, b):
        s_true, s_pred = float(np.sum(a)), float(np.sum(b))
        if s_pred <= 1e-6: return 1.0
        return float(np.clip(s_true/s_pred, CAL_CLIP[0], CAL_CLIP[1]))

    global_a = ratio(df["y_true"], df["y_pred"])
    item_dow = df.groupby(["메뉴","dow"]).apply(lambda g: ratio(g["y_true"], g["y_pred"]) if (g["y_true"]>0).sum()>=3 else np.nan).dropna()
    item_a   = df.groupby("메뉴").apply(lambda g: ratio(g["y_true"], g["y_pred"]) if (g["y_true"]>0).sum()>=3 else np.nan).dropna()
    shop_dow = df.groupby(["영업장명","dow"]).apply(lambda g: ratio(g["y_true"], g["y_pred"]) if (g["y_true"]>0).sum()>=3 else np.nan).dropna()
    shop_a   = df.groupby("영업장명").apply(lambda g: ratio(g["y_true"], g["y_pred"]) if (g["y_true"]>0).sum()>=3 else np.nan).dropna()

    print(f"[CAL] global={global_a:.3f}, item×dow={len(item_dow)}, item={len(item_a)}, shop×dow={len(shop_dow)}, shop={len(shop_a)}")
    return {
        "global": global_a,
        "item_dow": {(k[0], int(k[1])): float(v) for k,v in item_dow.items()},
        "item": item_a.to_dict(),
        "shop_dow": {(k[0], int(k[1])): float(v) for k,v in shop_dow.items()},
        "shop": shop_a.to_dict(),
    }

def apply_calibration_ratio(yhat, item_name, shop_name, dow, cal):
    a = cal["item_dow"].get((item_name, int(dow)))
    if a is None: a = cal["item"].get(item_name)
    if a is None: a = cal["shop_dow"].get((shop_name, int(dow)))
    if a is None: a = cal["shop"].get(shop_name, cal["global"])
    return float(yhat * a)

# ====================== Caps ======================
def compute_item_caps(train_df):
    # 최근 CAP_LOOKBACK일 기준 p95 * CAP_MULT
    tdf = train_df.copy()
    tdf["영업일자"] = pd.to_datetime(tdf["영업일자"], errors="coerce")
    last_date = tdf["영업일자"].max()
    start = last_date - pd.Timedelta(days=CAP_LOOKBACK-1)
    m = tdf["영업일자"] >= start
    grp = tdf.loc[m].groupby("영업장명_메뉴명")["매출수량"]
    caps = grp.quantile(CAP_P/100.0).fillna(0.0) * CAP_MULT
    return caps.to_dict()

# ====================== Train (Dual Models + Bagging Blend) ======================
def train_models_and_blend(train_df: pd.DataFrame):
    df = train_df.copy()
    assert {"영업일자","영업장명_메뉴명","매출수량"}.issubset(df.columns)
    df["영업일자"] = pd.to_datetime(df["영업일자"], errors="coerce")
    df = ensure_upjang(df)
    df["매출수량"] = pd.to_numeric(df["매출수량"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df = df.sort_values(["영업장명_메뉴명","영업일자"]).reset_index(drop=True)

    feat_df = build_features(df)
    feats = feature_columns(feat_df)
    y = feat_df["매출수량"].astype(float).values

    # 검증 분할
    max_date = feat_df["영업일자"].max()
    vstart = max_date - pd.Timedelta(days=VALID_LAST_DAYS-1)
    vmask = (feat_df["영업일자"] >= vstart)
    tmask = ~vmask

    X_tr, X_va = feat_df.loc[tmask, feats], feat_df.loc[vmask, feats]
    y_tr, y_va = y[tmask], y[vmask]
    shops_tr   = feat_df.loc[tmask, "영업장명"].astype(str).values
    shops_va   = feat_df.loc[vmask, "영업장명"].astype(str).values
    items_va   = feat_df.loc[vmask, "영업장명_메뉴명"].astype(str).values
    dows_va    = feat_df.loc[vmask, "dow"].astype(int).values

    # 가중치: 업장 가중 × 양성 부스트
    w_shop_tr = pd.Series(shops_tr).map(lambda s: SHOP_WEIGHTS.get(s,1.0)).values
    w_pos_tr  = np.where(y_tr>0, POS_SAMPLE_BOOST, 1.0)
    w_tr      = w_shop_tr * w_pos_tr
    w_va      = pd.Series(shops_va).map(lambda s: SHOP_WEIGHTS.get(s,1.0)).values

    # 배깅 컨테이너
    bag = []   # [(modelA, modelB, alpha, weight_inv_smape)]
    for seed in SEEDS:
        # ---- Model A: Tweedie (all rows) ----
        paramsA = dict(
            objective="reg:tweedie",
            tweedie_variance_power=1.2,
            eval_metric="rmse",
            tree_method="hist",
            eta=0.05,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=8,
            gamma=0.0,
            reg_lambda=2.5,
            reg_alpha=0.0,
            seed=seed,
            nthread=os.cpu_count() or 4,
            max_delta_step=1.0
        )
        dtrA = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        dvaA = xgb.DMatrix(X_va, label=y_va, weight=w_va)
        modelA = xgb.train(paramsA, dtrA, num_boost_round=3000,
                           evals=[(dvaA,"valid")], early_stopping_rounds=200, verbose_eval=False)
        predA_va = np.clip(modelA.predict(dvaA, iteration_range=(0, modelA.best_iteration+1)), 0, None)

        # ---- Model B: log1p-SSE (positives only) ----
        m_tr = y_tr > 0
        X_tr_pos, y_tr_pos = X_tr.loc[m_tr], np.log1p(y_tr[m_tr])
        w_tr_pos = w_tr[m_tr]
        paramsB = dict(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            eta=0.05,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=6,
            gamma=0.0,
            reg_lambda=2.0,
            reg_alpha=0.0,
            seed=seed,
            nthread=os.cpu_count() or 4
        )
        dtrB = xgb.DMatrix(X_tr_pos, label=y_tr_pos, weight=w_tr_pos)
        dvaB = xgb.DMatrix(X_va)
        modelB = xgb.train(paramsB, dtrB, num_boost_round=2000,
                           evals=[], early_stopping_rounds=None, verbose_eval=False)
        predB_va = np.expm1(np.clip(modelB.predict(dvaB), 0, None))

        # ---- Blend α search on validation ----
        alphas = np.linspace(0.1, 0.9, 17)
        best_alpha, best_score = 0.5, 1e9
        for a in alphas:
            pv = (1.0-a)*predA_va + a*predB_va
            pack = pd.DataFrame({
                "영업장명": shops_va,
                "영업장명_메뉴명": items_va,
                "영업일자": feat_df.loc[vmask, "영업일자"].values,
                "y_true": y_va, "y_pred": pv
            })
            sc = weighted_smape_like_official(pack)
            if sc < best_score:
                best_score, best_alpha = sc, a

        weight = 1.0 / max(best_score, 1e-6)  # 역SMAPE 가중
        bag.append((modelA, modelB, best_alpha, weight))
        print(f"[SEED {seed}] alpha={best_alpha:.3f} | valid Weighted SMAPE={best_score:.4f}")

    # 캘리브레이션(배깅 블렌드 기준)
    # 배깅 검증 예측을 가중 평균하여 캘리브 계산
    dva = xgb.DMatrix(X_va)
    y_pred_bag = np.zeros_like(y_va, dtype=float)
    wsum = 0.0
    for (mA, mB, a, w) in bag:
        pA = np.clip(mA.predict(dva, iteration_range=(0, mA.best_iteration+1)), 0, None)
        pB = np.expm1(np.clip(mB.predict(dva), 0, None))
        y_pred_bag += w * ((1.0-a)*pA + a*pB)
        wsum += w
    y_pred_bag /= max(wsum, 1e-6)

    cal = compute_calibration_from_validation(y_va, y_pred_bag, shops_va, items_va, dows_va)

    # 카테고리 매핑 & 캡
    cat_maps = {
        "item_to_id": pd.Categorical(feat_df["영업장명_메뉴명"].astype(str)).categories.tolist(),
        "upjang_to_id": pd.Categorical(feat_df["영업장명"].astype(str)).categories.tolist(),
    }
    caps = compute_item_caps(df) if ENABLE_CAP else {}
    return bag, feats, cat_maps, cal, caps

# ====================== 7일 재귀예측 ======================
def prepare_maps_from_lists(cat_maps):
    item_to_id_map = {name: i for i, name in enumerate(cat_maps["item_to_id"])}
    upjang_to_id_map = {name: i for i, name in enumerate(cat_maps["upjang_to_id"])}
    return item_to_id_map, upjang_to_id_map

def build_single_row_features(dt, cur_hist, item_id, upjang_id, bt_enc):
    dow = dt.weekday()
    week = int(dt.isocalendar().week)
    month, year, day = dt.month, dt.year, dt.day
    woy = int(dt.isocalendar().week)

    def pull(date):
        v = cur_hist.get(date, 0.0)
        if isinstance(v, pd.Series): v = float(v.sum())
        return float(v)

    def window_vals(win):
        return [pull(dt - timedelta(days=n)) for n in range(1, win+1)]
    def lag(n): return pull(dt - timedelta(days=n))

    vals7, vals14, vals28 = window_vals(7), window_vals(14), window_vals(28)
    lags = {f"lag_{n}": lag(n) for n in [1,7,14,28]}

    def roll_stats(vals):
        s = pd.Series(vals)
        if s.dropna().size >= 3:
            return float(s.mean()), float(s.std(ddof=0)), float(s.max()), float(s.min())
        return 0.0, 0.0, 0.0, 0.0

    rm7, rs7, rmax7, rmin7   = roll_stats(vals7)
    rm14, rs14, rmax14, rmin14 = roll_stats(vals14)
    rm28, rs28, rmax28, rmin28 = roll_stats(vals28)
    nz28 = float(np.mean(np.array(vals28) > 0)) if len(vals28) >= 3 else 0.0

    dlast = 365.0
    for n, v in enumerate(vals28, start=1):
        if v > 0: dlast = float(n); break

    is_weekend = 1 if dow >= 5 else 0

    # Expanding encodings은 예측 시 직접 계산 불가 → rolling 기반 간접 대치
    item_dow_idx = 1.0
    shop_dow_idx = 1.0

    return {
        "item_id": item_id, "업장_id": upjang_id,
        "dow": dow, "week": week, "month": month, "year": year, "day": day,
        "woy": woy,
        "woy_sin": np.sin(2*np.pi*woy/53), "woy_cos": np.cos(2*np.pi*woy/53),
        "roll_mean_7": rm7, "roll_std_7": rs7, "roll_max_7": rmax7, "roll_min_7": rmin7,
        "roll_mean_14": rm14, "roll_std_14": rs14, "roll_max_14": rmax14, "roll_min_14": rmin14,
        "roll_mean_28": rm28, "roll_std_28": rs28, "roll_max_28": rmax28, "roll_min_28": rmin28,
        **lags,
        "exp_item_dow_mean": rm7,
        "exp_item_mean": rm28,
        "exp_shop_dow_mean": rm7,
        "item_dow_idx": item_dow_idx,
        "shop_dow_idx": shop_dow_idx,
        "trend_7_1": rm7 / (lags["lag_1"] + 1e-6),
        "trend_14_7": rm14 / (rm7 + 1e-6),
        "delta_1_7": lags["lag_1"] - rm7,
        "nonzero_rate_28": nz28,
        "days_since_last_sale": dlast,
        "dow_sin": np.sin(2*np.pi*dow/7), "dow_cos": np.cos(2*np.pi*dow/7),
        "month_sin": np.sin(2*np.pi*month/12), "month_cos": np.cos(2*np.pi*month/12),
        "is_spike": 0, "is_drop": 0,
        "is_weekday_price": 1-is_weekend, "is_weekend_price": is_weekend,
        "is_holiday": 0,
        "_banquet_type_enc": bt_enc,
    }

def apply_peak_postprocess(yhat, dt, feat_row):
    if yhat < SOFT_ZERO_CUT: yhat = 0.0
    if dt.weekday() >= 5: yhat *= WEEKEND_BUMP
    if feat_row.get("is_holiday", 0) == 1: yhat *= HOLIDAY_BUMP
    try:
        if feat_row["lag_1"] > SPIKE_RATIO * (feat_row["roll_mean_7"] + 1e-6): yhat *= SPIKE_BUMP
    except Exception: pass
    return float(max(0.0, yhat))

def forecast_7days_for_testfile(bag, feats, train_df, test_df, test_tag, cat_maps, cal, caps):
    item_to_id_map, upjang_to_id_map = prepare_maps_from_lists(cat_maps)

    test_df = ensure_upjang(test_df.copy())
    test_df["영업일자"] = pd.to_datetime(test_df["영업일자"], errors="coerce")
    test_df = test_df.sort_values(["영업장명_메뉴명","영업일자"])
    last_date = test_df["영업일자"].max()
    items = sorted(test_df["영업장명_메뉴명"].astype(str).unique().tolist())

    cols = ["영업일자","영업장명_메뉴명","매출수량","banquet_type","영업장명"]
    a = ensure_upjang(train_df.copy()); b = ensure_upjang(test_df.copy())
    take = lambda df: df[[c for c in cols if c in df.columns]]
    hist_src = pd.concat([take(a), take(b)], ignore_index=True)
    hist_src["영업일자"] = pd.to_datetime(hist_src["영업일자"], errors="coerce")

    results = {k: {} for k in range(1,8)}
    for item in items:
        g = hist_src[hist_src["영업장명_메뉴명"].astype(str) == item].copy().sort_values("영업일자")
        y_series = (g.groupby("영업일자", as_index=True)["매출수량"].sum().astype(float).clip(lower=0.0))

        upjang = g["영업장명"].dropna().astype(str).iloc[-1] if "영업장명" in g.columns and len(g["영업장명"].dropna()) else (item.split("_",1)[0] if "_" in item else "")
        item_id  = item_to_id_map.get(item, -1)
        upjang_id = upjang_to_id_map.get(upjang, -1)
        bt_enc = -1
        cap_v = caps.get(item, np.inf)

        cur_hist = y_series.copy()
        for k in range(1,8):
            dt = last_date + timedelta(days=k)
            feat_row = build_single_row_features(dt, cur_hist, item_id, upjang_id, bt_enc)
            X = pd.DataFrame([feat_row])[feats].fillna(0.0)
            dmat = xgb.DMatrix(X)

            # 배깅 가중 평균
            yhat_sum, wsum = 0.0, 0.0
            for (mA, mB, a, w) in bag:
                yA = float(np.clip(mA.predict(dmat, iteration_range=(0, mA.best_iteration+1))[0], 0, None))
                yB = float(np.expm1(np.clip(mB.predict(dmat)[0], 0, None)))
                yhat = (1.0 - a) * yA + a * yB
                yhat_sum += w * yhat
                wsum += w
            yhat = yhat_sum / max(wsum, 1e-6)

            # 캘리브레이션 (item×dow → item → shop×dow → shop → global)
            yhat = apply_calibration_ratio(yhat, item, upjang, feat_row["dow"], cal)

            # 희소아이템 블렌딩
            nz   = float(feat_row.get("nonzero_rate_28", 0.0))
            dsl  = float(feat_row.get("days_since_last_sale", 999.0))
            lag7 = float(feat_row.get("lag_7", 0.0))
            rm7  = float(feat_row.get("roll_mean_7", 0.0))
            anchor = max(lag7, rm7)
            if (nz < SPARSE_NZ_THRESHOLD) or (dsl > STALE_DAYS_THRESHOLD):
                yhat = BLEND_MODEL * yhat + BLEND_ANCHOR_RATIO * anchor

            # 아이템 상한 클립(스파이크 컷)
            if ENABLE_CAP and np.isfinite(cap_v):
                yhat = min(yhat, float(cap_v))

            # 피크/주말 보정
            yhat = apply_peak_postprocess(yhat, dt, feat_row)

            results[k][item] = yhat
            cur_hist.loc[dt] = yhat  # 재귀 갱신

    out_rows = []
    for k in range(1,8):
        row = {"영업일자": f"{test_tag}+{k}일"}; row.update(results[k]); out_rows.append(row)
    return pd.DataFrame(out_rows).set_index("영업일자")

# ====================== 제출 저장(퍼지 매칭) ======================
def parse_tag_day(label: str):
    if not isinstance(label, str): label = str(label)
    m_tag = re.search(r"(TEST_\d{2})", label, flags=re.IGNORECASE)
    m_day = re.findall(r"(\d+)", label)
    tag = m_tag.group(1).upper() if m_tag else None
    k   = int(m_day[-1]) if m_day else None
    return tag, k

def norm_name(s: str):
    if not isinstance(s, str): s = str(s)
    s = unicodedata.normalize("NFKC", s).strip().lower()
    repl = {"·":" ","•":" ","ㆍ":" ","‧":" ","–":"-","—":"-","’":"'", "“":'"', "”":'"',
            "（":"(","）":")","【":"[","】":"]"}
    for k,v in repl.items(): s = s.replace(k, v)
    s = re.sub(r"[()\[\]{}]", " ", s)
    s = re.sub(r"[\/_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_row_index_map(pred_full: pd.DataFrame):
    pred_index_map = {}
    if pred_full is not None and not pred_full.empty:
        for ridx, row in pred_full.iterrows():
            t, k = parse_tag_day(ridx)
            if (t is not None) and (k is not None):
                pred_index_map[(t, k)] = row
    return pred_index_map

def build_column_mapping(sample_cols, pred_cols, cutoff=0.90):
    pred_norm2orig = {}
    for c in pred_cols:
        nc = norm_name(c)
        if nc not in pred_norm2orig: pred_norm2orig[nc] = c
    mapping, exact, fuzzy, unmatched = {}, 0, 0, []
    pred_norm_keys = list(pred_norm2orig.keys())
    for sc in sample_cols:
        ns = norm_name(sc)
        if ns in pred_norm2orig:
            mapping[sc] = pred_norm2orig[ns]; exact += 1
        else:
            cand = difflib.get_close_matches(ns, pred_norm_keys, n=1, cutoff=cutoff)
            if cand: mapping[sc] = pred_norm2orig[cand[0]]; fuzzy += 1
            else: mapping[sc] = None; unmatched.append(sc)
    return mapping, exact, fuzzy, unmatched

def save_submission(sample: pd.DataFrame, pred_full: pd.DataFrame, out_path: Path):
    pred_index_map = build_row_index_map(pred_full)
    submission = sample.copy()
    idx_labels = submission["영업일자"].tolist()
    item_cols  = submission.columns.tolist()[1:]

    pred_cols = [] if pred_full is None or pred_full.empty else pred_full.columns.tolist()
    col_map, exact, fuzzy, unmatch = build_column_mapping(item_cols, pred_cols, cutoff=0.90)

    out_vals, matched_rows = [], 0
    for lbl in idx_labels:
        t, k = parse_tag_day(lbl)
        if (t, k) in pred_index_map:
            sr = pred_index_map[(t, k)]
            row_vals = []
            for sc in item_cols:
                pc = col_map.get(sc)
                v = float(sr.get(pc, 0.0)) if pc is not None else 0.0
                row_vals.append(float(max(0.0, v)))
            matched_rows += 1
        else:
            row_vals = [0.0]*len(item_cols)
        out_vals.append(row_vals)

    final_df = pd.DataFrame(out_vals, columns=item_cols)
    final_df[item_cols] = np.round(final_df[item_cols].values, 0)
    final_df.insert(0, "영업일자", idx_labels)
    final_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    total_pred_sum = float(np.nansum(final_df[item_cols].to_numpy()))
    print(f"[DBG] matched_rows={matched_rows}/{len(idx_labels)} | exact={exact} fuzzy={fuzzy} unmatch={len(unmatch)} | nonzero_sum={total_pred_sum:.2f}")
    if unmatch[:5]: print("[DBG] sample columns not matched (first 5):", unmatch[:5])
    print(f"[OK] Saved submission: {out_path}  exists={os.path.exists(out_path)}")

# ====================== Main ======================
def main():
    train = safe_read_csv(TRAIN_FILE)
    sample = safe_read_csv(SAMPLE_SUB)

    train["영업일자"] = pd.to_datetime(train["영업일자"], errors="coerce")
    train = ensure_upjang(train).sort_values(["영업장명_메뉴명","영업일자"]).reset_index(drop=True)

    bag, feats, cat_maps, cal, caps = train_models_and_blend(train)   # → seed별 α/valid 로그 출력

    test_files = sorted(glob.glob(TEST_GLOB))
    print(f"[Info] Found {len(test_files)} test files: {test_files[:3]}{' ...' if len(test_files)>3 else ''}")

    all_pred_wide = []
    for tf in test_files:
        test_df = safe_read_csv(Path(tf))
        assert {"영업일자","영업장명_메뉴명","매출수량"}.issubset(test_df.columns), f"필수 컬럼 누락: {tf}"
        tag = Path(tf).stem.split("_")[1]   # TEST_00_processed -> '00'
        test_tag = f"TEST_{tag}"
        wide = forecast_7days_for_testfile(bag, feats, train, test_df, test_tag, cat_maps, cal, caps)
        all_pred_wide.append(wide)

    pred_full = pd.concat(all_pred_wide, axis=0) if all_pred_wide else pd.DataFrame()
    save_submission(sample, pred_full, OUT_FILE)

if __name__ == "__main__":
    main()
