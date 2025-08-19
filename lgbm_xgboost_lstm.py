# -*- coding: utf-8 -*-
"""
Resort F&B — 7-Day Demand Forecast (Triple Ensemble: XGB + LGBM + LSTM)
- 시간누수 방지 피처 + 업장가중 sMAPE 검증(Last 28D)
- grid로 앙상블 가중 탐색
- 다단 캘리브레이션(item×dow > item > shop×dow > shop > global)
- 희소 블렌딩 + item p98 상한 + 주말/피크 소프트 보정
- 0 금지: 예측 하한(0.51) + 제출하한(정수 1)
"""

import os, glob, warnings
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd

# ====================== 0) PATHS ======================
BASE_DIR   = Path(r"C:\Users\minseo\lg")  # ▶▶ 본인 PC 경로로 조정
TRAIN_PATH = BASE_DIR / "train_01" / "re_train_06.csv"
TEST_GLOB  = str(BASE_DIR / "test_01" / "TEST_*_processed.csv")
SAMPLE_SUB = BASE_DIR / "sample_submission.csv"
OUT_PATH   = BASE_DIR / "submission_triple_ens_nozero.csv"

# ====================== 1) CONFIG ======================
RANDOM_STATE = 42
VALID_LAST_DAYS = 28
SEQ_LEN = 28
EPOCHS  = 12
BATCH   = 512

SHOP_WEIGHTS = {"담하": 2.0, "미라시아": 2.0}  # 그 외 1.0

SPARSE_NZ_THRESHOLD = 0.06
STALE_DAYS_THRESHOLD = 28
BLEND_MODEL = 0.72
BLEND_ANCHOR_RATIO = 0.28

CAP_LOOKBACK = 56
CAP_P = 98
CAP_MULT = 1.25
CAP_MIN_APPLY = 4.0
CAP_MIN_NZ28  = 0.60
ENABLE_CAP = True

WEEKEND_BUMP = 1.05
SPIKE_BUMP   = 1.06
SPIKE_RATIO  = 1.50

MIN_PRED_BEFORE_ROUND = 0.51
MIN_SUBMISSION_INT    = 1

np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore")

# ====================== 2) IO & UTILS ======================
def safe_read_csv(path: Path):
    for enc in ("utf-8-sig","utf-8","cp949","euc-kr"):
        try: return pd.read_csv(path, encoding=enc)
        except Exception: pass
    raise RuntimeError(f"CSV 로드 실패: {path}")

def ensure_cols(df):
    need = {"영업일자","영업장명_메뉴명","매출수량"}
    if not need.issubset(df.columns):
        raise ValueError(f"필수 컬럼 누락: {list(need-set(df.columns))}")
    return df

def ensure_upjang(df):
    if "영업장명" not in df.columns:
        df["영업장명"] = df["영업장명_메뉴명"].astype(str).str.split("_",1).str[0]
    return df

def smape_ignore_zero(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = y_true != 0
    if m.sum() == 0: return 0.0
    yt, yp = y_true[m], y_pred[m]
    return 100.0 * np.mean(2.0*np.abs(yp-yt)/(np.abs(yt)+np.abs(yp)+eps))

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

# ====================== 3) FEATURES ======================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["영업일자"] = pd.to_datetime(out["영업일자"], errors="coerce")
    out = ensure_upjang(out)
    out["매출수량"] = pd.to_numeric(out["매출수량"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out = out.sort_values(["영업장명_메뉴명","영업일자"]).reset_index(drop=True)

    out["dow"]   = out["영업일자"].dt.weekday
    out["month"] = out["영업일자"].dt.month
    out["woy"]   = out["영업일자"].dt.isocalendar().week.astype(int)
    out["day_of_year"] = out["영업일자"].dt.dayofyear

    for lag in [1,7,14,28]:
        out[f"lag_{lag}"] = out.groupby("영업장명_메뉴명")["매출수량"].shift(lag)

    for win in [7,14,28]:
        grp = out.groupby("영업장명_메뉴명")["매출수량"]
        out[f"roll_mean_{win}"] = grp.shift(1).rolling(win, min_periods=3).mean()
        out[f"roll_std_{win}"]  = grp.shift(1).rolling(win, min_periods=3).std()
        out[f"roll_max_{win}"]  = grp.shift(1).rolling(win, min_periods=3).max()
        out[f"roll_min_{win}"]  = grp.shift(1).rolling(win, min_periods=3).min()

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
    out["item_dow_idx"] = out["exp_item_dow_mean"] / (out["exp_item_mean"] + 1e-6)

    out["trend_7_1"]  = out["roll_mean_7"]  / (out["lag_1"] + 1e-6)
    out["trend_14_7"] = out["roll_mean_14"] / (out["roll_mean_7"] + 1e-6)
    out["delta_1_7"]  = out["lag_1"] - out["roll_mean_7"]

    out["nonzero_rate_28"] = (
        out.groupby("영업장명_메뉴명")["매출수량"]
          .transform(lambda s: s.shift(1).rolling(28, min_periods=3).apply(lambda x: (x>0).mean(), raw=True))
    )
    def dsl(g):
        last=None; res=[]
        for d,y in zip(g["영업일자"], g["매출수량"]):
            res.append(365 if last is None else (d-last).days)
            if y>0: last=d
        return pd.Series(res, index=g.index)
    out["days_since_last_sale"] = out.groupby("영업장명_메뉴명", group_keys=False).apply(dsl).astype(float)

    out["dow_sin"] = np.sin(2*np.pi*out["dow"]/7)
    out["dow_cos"] = np.cos(2*np.pi*out["dow"]/7)
    out["woy_sin"] = np.sin(2*np.pi*out["woy"]/53)
    out["woy_cos"] = np.cos(2*np.pi*out["woy"]/53)
    out["doy_sin"] = np.sin(2*np.pi*out["day_of_year"]/365.25)
    out["doy_cos"] = np.cos(2*np.pi*out["day_of_year"]/365.25)

    out["item_id"] = out["영업장명_메뉴명"].astype("category").cat.codes
    out["업장_id"]  = out["영업장명"].astype("category").cat.codes

    num = out.select_dtypes(include=[np.number]).columns.tolist()
    if "매출수량" in num: num.remove("매출수량")
    out[num] = out[num].fillna(0.0)
    return out

def feature_columns(df: pd.DataFrame):
    base = [
        "item_id","업장_id","dow","month","woy","day_of_year",
        "woy_sin","woy_cos","doy_sin","doy_cos",
        "lag_1","lag_7","lag_14","lag_28",
        "roll_mean_7","roll_std_7","roll_max_7","roll_min_7",
        "roll_mean_14","roll_std_14","roll_max_14","roll_min_14",
        "roll_mean_28","roll_std_28","roll_max_28","roll_min_28",
        "exp_item_dow_mean","exp_item_mean","item_dow_idx",
        "trend_7_1","trend_14_7","delta_1_7",
        "nonzero_rate_28","days_since_last_sale",
        "dow_sin","dow_cos"
    ]
    return [c for c in base if c in df.columns]

# ====================== 4) CALIBRATION & CAPS ======================
CAL_CLIP = (0.80, 1.20)
CAL_MIN_ITEM_DOW = 4
CAL_MIN_ITEM     = 4
CAL_MIN_SHOP_DOW = 5
CAL_MIN_SHOP     = 5

def compute_calibration_from_validation(y_true, y_pred, shops, items, dows):
    df = pd.DataFrame({"y_true":y_true,"y_pred":y_pred,"영업장명":shops,"메뉴":items,"dow":dows})
    df = df[df["y_true"]>0]
    def ratio(a,b):
        s_true, s_pred = float(np.sum(a)), float(np.sum(b))
        if s_pred <= 1e-6: return 1.0
        return float(np.clip(s_true/s_pred, CAL_CLIP[0], CAL_CLIP[1]))
    def grp_ratio(g, min_nz):
        return ratio(g["y_true"], g["y_pred"]) if (g["y_true"]>0).sum() >= min_nz else np.nan
    global_a = ratio(df["y_true"], df["y_pred"])
    item_dow = (df.groupby(["메뉴","dow"]).apply(lambda g: grp_ratio(g, CAL_MIN_ITEM_DOW)).dropna())
    item_a   = (df.groupby("메뉴").apply(lambda g: grp_ratio(g, CAL_MIN_ITEM)).dropna())
    shop_dow = (df.groupby(["영업장명","dow"]).apply(lambda g: grp_ratio(g, CAL_MIN_SHOP_DOW)).dropna())
    shop_a   = (df.groupby("영업장명").apply(lambda g: grp_ratio(g, CAL_MIN_SHOP)).dropna())
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

def compute_item_caps(train_df):
    tdf = train_df.copy()
    tdf["영업일자"] = pd.to_datetime(tdf["영업일자"], errors="coerce")
    last_date = tdf["영업일자"].max()
    start = last_date - pd.Timedelta(days=CAP_LOOKBACK-1)
    m = tdf["영업일자"] >= start
    grp = tdf.loc[m].groupby("영업장명_메뉴명")["매출수량"]
    caps = grp.quantile(CAP_P/100.0).fillna(0.0) * CAP_MULT
    caps = caps[caps >= CAP_MIN_APPLY]
    nz28 = (tdf.loc[m]
              .groupby("영업장명_메뉴명")["매출수량"]
              .apply(lambda s: (s.tail(28) > 0).mean() if len(s) >= 1 else 0.0))
    eligible = nz28[nz28 >= CAP_MIN_NZ28].index
    caps = caps[caps.index.isin(eligible)]
    return caps.to_dict()

def compute_item_dow_factors(y_series: pd.Series, window_days=90, min_count=3, clip_lo=0.75, clip_hi=1.35):
    if y_series is None or len(y_series) == 0:
        return {d:1.0 for d in range(7)}
    end = y_series.index.max()
    start = end - pd.Timedelta(days=window_days-1)
    s = y_series.loc[y_series.index >= start]
    if s.empty: return {d:1.0 for d in range(7)}
    df = pd.DataFrame({"y": s}); df["dow"] = df.index.weekday
    overall = df["y"].mean()
    if not np.isfinite(overall) or overall <= 0:
        overall = float(s.rolling(28, min_periods=1).mean().iloc[-1] or 1.0)
    fac = {}
    for d in range(7):
        ys = df.loc[df["dow"]==d, "y"]
        if (ys>0).sum() >= min_count:
            val = float(np.clip(ys.mean()/(overall+1e-6), clip_lo, clip_hi))
        else: val = 1.0
        fac[d]=val
    return fac

# ====================== 5) MODELS ======================
xgb_ok = True
try:
    import xgboost as xgb
except Exception:
    xgb_ok = False

lgb_ok = True
try:
    import lightgbm as lgb
except Exception:
    lgb_ok = False
from sklearn.ensemble import GradientBoostingRegressor

tf_ok = True
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    tf_ok = False

def train_xgb(X_tr, y_tr, X_va, y_va, w_tr=None):
    """XGB(log1p positives) — valid DMatrix에 label 포함"""
    if not xgb_ok: return None
    m_tr = y_tr > 0
    if m_tr.sum() == 0:
        def predict(X): return np.zeros(len(X), dtype=float)
        return predict
    dtr = xgb.DMatrix(X_tr.loc[m_tr], label=np.log1p(y_tr[m_tr]), weight=(w_tr[m_tr] if w_tr is not None else None))
    use_valid = (len(X_va) > 0) and (len(y_va) == len(X_va))
    evals = []
    if use_valid:
        dva = xgb.DMatrix(X_va, label=np.log1p(np.clip(y_va, 0, None)))
        evals = [(dva, "valid")]
    params = dict(objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
                  eta=0.05, max_depth=8, subsample=0.9, colsample_bytree=0.9,
                  min_child_weight=6, reg_lambda=2.0, seed=RANDOM_STATE)
    model = xgb.train(params, dtr, num_boost_round=2000, verbose_eval=False,
                      evals=evals, early_stopping_rounds=(200 if evals else None))
    best_it = getattr(model, "best_iteration", None)
    def predict(X):
        dm = xgb.DMatrix(X)
        if best_it is not None:
            preds = model.predict(dm, iteration_range=(0, best_it+1))
        else:
            preds = model.predict(dm)
        return np.expm1(np.clip(preds, 0, None))
    return predict

def train_lgbm(X_tr, y_tr, X_va, y_va, w_tr=None):
    """LightGBM Tweedie — 콜백 기반 early stopping(버전 호환)"""
    if lgb_ok:
        dtr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        valid_sets = []
        callbacks = [lgb.log_evaluation(0)]  # 조용히
        if len(X_va) > 0 and len(y_va) == len(X_va):
            dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
            valid_sets = [dva]
            callbacks.append(lgb.early_stopping(stopping_rounds=300, verbose=False))
        params = dict(objective="tweedie", tweedie_variance_power=1.2,
                      metric="rmse", learning_rate=0.05, num_leaves=63,
                      feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                      min_data_in_leaf=40, reg_lambda=2.0, seed=RANDOM_STATE)
        model = lgb.train(params, dtr,
                          valid_sets=valid_sets if valid_sets else [dtr],
                          num_boost_round=5000,
                          callbacks=callbacks)
        def predict(X):
            it = getattr(model, "best_iteration", None)
            return np.clip(model.predict(X, num_iteration=it), 0, None)
        return predict
    # fallback: GBRT on log1p (positives only)
    m_tr = y_tr > 0
    if m_tr.sum() == 0:
        def predict(X): return np.zeros(len(X), dtype=float)
        return predict
    gbr = GradientBoostingRegressor(learning_rate=0.05, n_estimators=800, max_depth=5,
                                    subsample=0.9, random_state=RANDOM_STATE)
    gbr.fit(X_tr.loc[m_tr], np.log1p(y_tr[m_tr]), sample_weight=(w_tr[m_tr] if w_tr is not None else None))
    def predict(X): return np.expm1(np.clip(gbr.predict(X), 0, None))
    return predict

def make_seq_matrix(series, end_dt, seq_len=SEQ_LEN):
    X = []
    for n in range(seq_len, 0, -1):
        d = end_dt - timedelta(days=n)
        y = float(series.get(d, 0.0))
        dow = d.weekday(); mon = d.month
        X.append([np.log1p(y),
                  np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
                  np.sin(2*np.pi*mon/12), np.cos(2*np.pi*mon/12)])
    return np.array(X, float)

def train_lstm(train_df, valid_end):
    if not tf_ok: return None
    seqs, targets = [], []
    for _, g in train_df.groupby("영업장명_메뉴명"):
        s = (g.groupby("영업일자")["매출수량"].sum().astype(float)).sort_index()
        dates = s.index.to_list()
        for i in range(SEQ_LEN, len(dates)):
            dt = dates[i]
            if dt > valid_end: break
            x = make_seq_matrix(s, dt- pd.Timedelta(days=1), SEQ_LEN)
            y = np.log1p(float(s.loc[dt]))
            seqs.append(x); targets.append(y)
    if not seqs: return None
    X = np.stack(seqs); y = np.array(targets, float)
    inp = keras.Input(shape=(SEQ_LEN, 5))
    x = keras.layers.Masking(mask_value=0.0)(inp)
    x = keras.layers.LSTM(64)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    out = keras.layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    es = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")
    model.fit(X, y, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH, verbose=0, callbacks=[es])

    def predict_one_step(series, target_dt):
        x = make_seq_matrix(series, target_dt- pd.Timedelta(days=1), SEQ_LEN)
        x = x[None, ...]
        y = float(np.expm1(np.clip(model.predict(x, verbose=0)[0,0], 0, None)))
        return y
    return predict_one_step

# ====================== 6) TRAIN & WEIGHTS ======================
def train_all_and_validate(train_df):
    df = ensure_upjang(train_df.copy())
    df["영업일자"] = pd.to_datetime(df["영업일자"], errors="coerce")
    df["매출수량"] = pd.to_numeric(df["매출수량"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df = df.sort_values(["영업장명_메뉴명","영업일자"]).reset_index(drop=True)

    feat = build_features(df)
    feats = feature_columns(feat)
    y = feat["매출수량"].astype(float).values

    max_date = feat["영업일자"].max()
    vstart   = max_date - pd.Timedelta(days=VALID_LAST_DAYS-1)
    vmask    = (feat["영업일자"] >= vstart)
    tmask    = ~vmask

    X_tr, X_va = feat.loc[tmask, feats], feat.loc[vmask, feats]
    y_tr, y_va = y[tmask], y[vmask]
    shops_va   = feat.loc[vmask, "영업장명"].astype(str).values
    items_va   = feat.loc[vmask, "영업장명_메뉴명"].astype(str).values
    dows_va    = feat.loc[vmask, "dow"].astype(int).values

    shops_tr = feat.loc[tmask, "영업장명"].astype(str).values
    w_shop_tr = pd.Series(shops_tr).map(lambda s: SHOP_WEIGHTS.get(s,1.0)).values
    w_pos_tr  = np.where(y_tr>0, 4.0, 1.0)
    w_tr      = w_shop_tr * w_pos_tr

    pred_xgb  = train_xgb(X_tr, y_tr, X_va, y_va, w_tr)
    pred_lgb  = train_lgbm(X_tr, y_tr, X_va, y_va, w_tr)
    pred_lstm = train_lstm(df, max_date)

    px = pred_xgb(X_va)  if pred_xgb  else np.zeros_like(y_va)
    pl = pred_lgb(X_va)  if pred_lgb  else np.zeros_like(y_va)

    if pred_lstm:
        plstm = []
        for it, d in zip(items_va, feat.loc[vmask,"영업일자"].values):
            s = (df.loc[df["영업장명_메뉴명"]==it, ["영업일자","매출수량"]]
                   .groupby("영업일자")["매출수량"].sum().astype(float).sort_index())
            if len(s) < SEQ_LEN: plstm.append(0.0)
            else: plstm.append(pred_lstm(s, d))
        plstm = np.array(plstm, float)
    else:
        plstm = np.zeros_like(y_va)

    best = (None, 1e9)
    grid = np.linspace(0,1,11)
    for wx in grid:
        for wl in grid:
            wlstm = 1.0 - wx - wl
            if wlstm < 0 or wlstm > 1: continue
            pv = wx*px + wl*pl + wlstm*plstm
            pack = pd.DataFrame({"영업장명":shops_va,"영업장명_메뉴명":items_va,
                                 "y_true":y_va,"y_pred":pv})
            sc = weighted_smape_like_official(pack)
            if sc < best[1]: best = ((wx,wl,wlstm), sc)
    (wx, wl, wlstm), best_sc = best
    print(f"[VALID] xgb={wx:.2f}, lgb={wl:.2f}, lstm={wlstm:.2f} | sMAPE={best_sc:.4f}")

    cal  = compute_calibration_from_validation(y_va, wx*px+wl*pl+wlstm*plstm, shops_va, items_va, dows_va)
    caps = compute_item_caps(df) if ENABLE_CAP else {}
    cat_maps = {
        "item_to_id": pd.Categorical(feat["영업장명_메뉴명"].astype(str)).categories.tolist(),
        "upjang_to_id": pd.Categorical(feat["영업장명"].astype(str)).categories.tolist(),
    }
    models  = {"xgb": pred_xgb, "lgb": pred_lgb, "lstm": pred_lstm}
    weights = {"xgb": wx, "lgb": wl, "lstm": wlstm}
    return models, weights, feats, cat_maps, cal, caps, df

# ====================== 7) SINGLE ROW & FORECAST ======================
def prepare_maps_from_lists(cat_maps):
    return ({name:i for i,name in enumerate(cat_maps["item_to_id"])},
            {name:i for i,name in enumerate(cat_maps["upjang_to_id"])})

def build_single_row_features(dt, cur_hist, item_id, upjang_id):
    dow = dt.weekday(); month = dt.month
    woy = int(dt.isocalendar().week); doy = int(dt.timetuple().tm_yday)
    def pull(date): return float(cur_hist.get(date, 0.0))
    def window_vals(win): return [pull(dt - timedelta(days=n)) for n in range(1, win+1)]
    vals7, vals14, vals28 = window_vals(7), window_vals(14), window_vals(28)
    def roll_stats(vals):
        s = pd.Series(vals)
        return (float(s.mean() if len(vals) else 0.0),
                float(s.std(ddof=0) if len(vals) else 0.0),
                float(s.max() if len(vals) else 0.0),
                float(s.min() if len(vals) else 0.0))
    rm7, rs7, rmax7, rmin7     = roll_stats(vals7)
    rm14, rs14, rmax14, rmin14 = roll_stats(vals14)
    rm28, rs28, rmax28, rmin28 = roll_stats(vals28)
    nz28 = float(np.mean(np.array(vals28)>0)) if len(vals28) else 0.0
    dlast = 365.0
    for n,v in enumerate(vals28,1):
        if v>0: dlast=float(n); break

    return {
        "item_id": item_id, "업장_id": upjang_id,
        "dow":dow, "month":month, "woy":woy, "day_of_year":doy,
        "woy_sin":np.sin(2*np.pi*woy/53), "woy_cos":np.cos(2*np.pi*woy/53),
        "doy_sin":np.sin(2*np.pi*doy/365.25), "doy_cos":np.cos(2*np.pi*doy/365.25),
        "lag_1":pull(dt - timedelta(days=1)),
        "lag_7":pull(dt - timedelta(days=7)),
        "lag_14":pull(dt - timedelta(days=14)),
        "lag_28":pull(dt - timedelta(days=28)),
        "roll_mean_7":rm7,"roll_std_7":rs7,"roll_max_7":rmax7,"roll_min_7":rmin7,
        "roll_mean_14":rm14,"roll_std_14":rs14,"roll_max_14":rmax14,"roll_min_14":rmin14,
        "roll_mean_28":rm28,"roll_std_28":rs28,"roll_max_28":rmax28,"roll_min_28":rmin28,
        "exp_item_dow_mean":rm7, "exp_item_mean":rm28, "item_dow_idx":1.0,
        "trend_7_1": rm7/(pull(dt - timedelta(days=1))+1e-6),
        "trend_14_7": rm14/(rm7+1e-6),
        "delta_1_7": pull(dt - timedelta(days=1)) - rm7,
        "nonzero_rate_28": nz28, "days_since_last_sale": dlast,
        "dow_sin":np.sin(2*np.pi*dow/7), "dow_cos":np.cos(2*np.pi*dow/7)
    }

def forecast_7days_for_testfile(models, weights, feats, train_df, test_df, test_tag, cat_maps, cal, caps):
    item_map, upjang_map = prepare_maps_from_lists(cat_maps)
    test_df = ensure_upjang(test_df.copy())
    test_df["영업일자"] = pd.to_datetime(test_df["영업일자"], errors="coerce")
    test_df = test_df.sort_values(["영업장명_메뉴명","영업일자"])
    last_date = test_df["영업일자"].max()
    items = sorted(test_df["영업장명_메뉴명"].astype(str).unique().tolist())

    cols = ["영업일자","영업장명_메뉴명","매출수량","영업장명"]
    hist = pd.concat([ensure_upjang(train_df)[cols], ensure_upjang(test_df)[cols]], ignore_index=True)
    hist["영업일자"] = pd.to_datetime(hist["영업일자"], errors="coerce")

    results = {k:{} for k in range(1,8)}
    for item in items:
        g = hist[hist["영업장명_메뉴명"].astype(str)==item].copy().sort_values("영업일자")
        y_series = (g.groupby("영업일자", as_index=True)["매출수량"].sum().astype(float).clip(lower=0.0))
        upjang = g["영업장명"].dropna().astype(str).iloc[-1] if "영업장명" in g.columns and len(g["영업장명"].dropna()) else (item.split("_",1)[0] if "_" in item else "")
        item_id = item_map.get(item, -1); upjang_id = upjang_map.get(upjang, -1)

        factors = compute_item_dow_factors(y_series, 90, 3)
        cap_v = caps.get(item, np.inf)
        cur = y_series.copy()

        for k in range(1,8):
            dt = last_date + timedelta(days=k)
            frow = build_single_row_features(dt, cur, item_id, upjang_id)
            X = pd.DataFrame([frow])[feats].fillna(0.0)

            preds = []
            if models["xgb"]:  preds.append(weights["xgb"]  * float(models["xgb"](X)[0]))
            if models["lgb"]:  preds.append(weights["lgb"]  * float(models["lgb"](X)[0]))
            if models["lstm"] and len(cur) >= SEQ_LEN:
                preds.append(weights["lstm"] * float(models["lstm"](cur, dt)))
            yhat = float(sum(preds)) if preds else 0.0

            yhat = apply_calibration_ratio(yhat, item, upjang, frow["dow"], cal)
            yhat *= factors.get(int(frow["dow"]), 1.0)

            nz, dsl = float(frow["nonzero_rate_28"]), float(frow["days_since_last_sale"])
            anchor = max(float(frow["lag_7"]), float(frow["roll_mean_7"]))
            if (nz < SPARSE_NZ_THRESHOLD) or (dsl > STALE_DAYS_THRESHOLD):
                yhat = BLEND_MODEL * yhat + BLEND_ANCHOR_RATIO * anchor

            if ENABLE_CAP and np.isfinite(cap_v):
                if (frow["nonzero_rate_28"] >= CAP_MIN_NZ28) and (cap_v >= CAP_MIN_APPLY):
                    yhat = min(yhat, float(cap_v))

            if dt.weekday() >= 5: yhat *= WEEKEND_BUMP
            try:
                if frow["lag_1"] > SPIKE_RATIO*(frow["roll_mean_7"]+1e-6): yhat *= SPIKE_BUMP
            except Exception: pass

            yhat = max(MIN_PRED_BEFORE_ROUND, yhat)

            results[k][item] = yhat
            cur.loc[dt] = yhat

    out_rows = []
    for k in range(1,8):
        row = {"영업일자": f"{test_tag}+{k}일"}; row.update(results[k]); out_rows.append(row)
    return pd.DataFrame(out_rows).set_index("영업일자")

# ====================== 8) SUBMISSION (no zeros) ======================
def save_submission(sample: pd.DataFrame, pred_full: pd.DataFrame, out_path: Path):
    submission = sample.copy()
    item_cols = submission.columns.tolist()[1:]
    idx_labels = submission["영업일자"].tolist()

    wide = pred_full.reindex(idx_labels).fillna(MIN_PRED_BEFORE_ROUND)
    vals = []
    for lbl in idx_labels:
        if lbl in wide.index:
            sr = wide.loc[lbl]
            vals.append([float(sr.get(c, MIN_PRED_BEFORE_ROUND)) for c in item_cols])
        else:
            vals.append([MIN_PRED_BEFORE_ROUND]*len(item_cols))

    final_df = pd.DataFrame(vals, columns=item_cols)
    arr = np.rint(final_df[item_cols].to_numpy(float))
    arr[arr < MIN_SUBMISSION_INT] = MIN_SUBMISSION_INT
    final_df[item_cols] = arr.astype(int)
    final_df.insert(0, "영업일자", idx_labels)
    final_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    zeros = (final_df[item_cols].to_numpy(int)==0).sum()
    print(f"[OK] Saved: {out_path} | zeros={zeros}")

# ====================== 9) MAIN ======================
def main():
    train = ensure_cols(safe_read_csv(TRAIN_PATH))
    sample = safe_read_csv(SAMPLE_SUB)

    models, weights, feats, cat_maps, cal, caps, train_df = train_all_and_validate(train)

    all_pred = []
    for tf in sorted(glob.glob(TEST_GLOB)):
        tdf = ensure_cols(safe_read_csv(Path(tf)))
        tdf["영업일자"] = pd.to_datetime(tdf["영업일자"], errors="coerce")
        tdf["매출수량"] = pd.to_numeric(tdf["매출수량"], errors="coerce").fillna(0.0).clip(lower=0.0)
        tag = Path(tf).stem.split("_")[1]   # TEST_00_processed → '00'
        test_tag = f"TEST_{tag}"
        wide = forecast_7days_for_testfile(models, weights, feats, train_df, tdf, test_tag, cat_maps, cal, caps)
        all_pred.append(wide)

    pred_full = pd.concat(all_pred, axis=0) if all_pred else pd.DataFrame()
    save_submission(sample, pred_full, OUT_PATH)

if __name__ == "__main__":
    main()
