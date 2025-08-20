# -*- coding: utf-8 -*-
"""
LG Aimers — 7일 메뉴 수요예측 (GPU-강화 앙상블)
+ 사용자가 제공한 달력 플래그(is_holiday, is_before_holiday, is_after_holiday,
  is_sandwich, between_holidays, is_weekend) 학습/예측 피처에 반영

구조는 기존과 동일(트리: XGB/LGB + DL: CNN-LSTM + ZI 라우팅 + 가드레일 + 제출 CSV)

주의:
- 달력 플래그가 train/test에 들어있으면 그대로 사용합니다(권장).
- 미래 7일에 대해 플래그가 없다면 is_weekend만 요일로 자동 보정(토=5, 일=6).
- 나머지 휴일 관련 플래그가 미래 7일에 비어 있으면 0으로 둡니다(데이터 제공 권장).
"""

import os, glob, warnings, random
import numpy as np
import pandas as pd
from datetime import timedelta

warnings.filterwarnings("ignore")
np.random.seed(42); random.seed(42)

# ===== 경로(필요 시 사용자 환경에 맞게) =====
TRAIN_FILE =  r"C:\Users\system1\Downloads\open\re_data_processed\re_train_08.csv"
TEST_GLOB  =  r"C:\Users\system1\Downloads\open\re_data_processed\re_test_processed_06\TEST_*_processed.csv"
SAMPLE_SUB =  r"C:\Users\system1\Downloads\open\sample_submission.csv"
OUT_FILE   =  r"C:\Users\system1\Downloads\open\submission_cnn_lstm_xgb_lgbm_.csv"

TRAIN_CAND = [
    TRAIN_FILE,
    r"/mnt/data/re_train_07.csv",
    r"/mnt/data/re_train_06.csv",
    r"/mnt/data/train.csv",
]

# ===== 하이퍼파라미터 =====
VALID_RATIO     = 0.12
EARLY_STOP_XGB  = 200
EARLY_STOP_LGB  = 300
NUM_ROUNDS_XGB  = 5000
NUM_ROUNDS_LGB  = 8000

DL_WIN          = 35
DL_DROPOUT      = 0.2

# ZI 라우팅
ZI_ZERO_RATIO   = 0.65
ZI_CV           = 1.2
ZI_N_POS        = 6
ALPHA_MAX       = 0.45

# ===== 유틸 =====
def safe_read_csv(path):
    for enc in ("utf-8-sig","utf-8","cp949","euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"CSV 로드 실패: {path}")

def find_train_path():
    for p in TRAIN_CAND:
        if os.path.exists(p): return p
    raise FileNotFoundError("훈련 데이터 없음(re_train_07/06/train.csv)")

def ensure_basic_cols(df):
    for c in ["영업일자","영업장명_메뉴명","매출수량"]:
        if c not in df.columns: raise ValueError(f"필수 컬럼 누락: {c}")
    return df

def normalize_series_from_group(g):
    s = g.groupby(pd.to_datetime(g["영업일자"]).dt.normalize())["매출수량"].sum().astype(float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()

def pull_val(series, d):
    v = series.get(d, 0.0)
    if isinstance(v, pd.Series): v = v.sum()
    return float(v)

def list_test_files():
    files = sorted(glob.glob(TEST_GLOB))
    if not files:
        files = sorted(glob.glob("/mnt/data/TEST_*_processed.csv"))
    if not files:
        raise FileNotFoundError("TEST_*_processed.csv 경로를 찾을 수 없습니다.")
    return files

# ===== 달력 플래그 처리 =====
CAL_FLAG_COLS = [
    "is_holiday",
    "is_before_holiday",
    "is_after_holiday",
    "is_sandwich",
    "between_holidays",
    "is_weekend",
]

def build_calendar_flag_map(df):
    """
    날짜별 달력 플래그 딕셔너리 생성.
    - 입력 df에 위 CAL_FLAG_COLS 중 존재하는 컬럼만 사용(없으면 0으로 대체).
    - 같은 날짜 내 값이 다르면 평균→0/1로 반올림(일반적으로 동일해야 함).
    """
    dfx = df.copy()
    dfx["영업일자"] = pd.to_datetime(dfx["영업일자"]).dt.normalize()
    # 수치화
    for c in CAL_FLAG_COLS:
        if c in dfx.columns:
            dfx[c] = pd.to_numeric(dfx[c], errors="coerce").fillna(0.0).clip(0, 1)
    g = dfx.groupby("영업일자")
    cal_map = {}
    for dt, grp in g:
        flags = {}
        for c in CAL_FLAG_COLS:
            if c in grp.columns:
                v = float(np.rint(np.nanmean(pd.to_numeric(grp[c], errors="coerce").fillna(0).values)))
                flags[c] = int(max(0, min(1, v)))
        cal_map[pd.to_datetime(dt)] = flags
    return cal_map

def get_flags_for_date(cal_map, target_date):
    """
    특정 날짜의 달력 플래그 반환.
    - cal_map에 없으면 0으로 채움.
    - is_weekend은 최저 보정: 토(5)/일(6)이면 1로 세팅.
    """
    flags = {c:0 for c in CAL_FLAG_COLS}
    if target_date in cal_map:
        for k,v in cal_map[target_date].items():
            flags[k] = int(v)
    # 주말 보정(데이터가 비었을 때 최소한의 정보)
    if flags.get("is_weekend", 0) == 0:
        if target_date.weekday() in (5, 6):  # 토/일
            flags["is_weekend"] = 1
    return flags

# ===== 피처 생성 =====
def build_feats_from_series(values, target_date, cal_map):
    lag = lambda n: pull_val(values, target_date - timedelta(days=n))
    def roll(win):
        arr = np.array([pull_val(values, target_date - timedelta(days=n)) for n in range(1, win+1)], float)
        if arr.size == 0: return 0,0,0,0
        return arr.mean(), arr.std(ddof=0), arr.max(), arr.min()

    l1,l7,l14,l28 = [lag(n) for n in (1,7,14,28)]
    rm7,rs7,rmax7,rmin7   = roll(7)
    rm14,rs14,rmax14,rmin14 = roll(14)
    rm28,rs28,rmax28,rmin28 = roll(28)
    arr28 = [pull_val(values, target_date - timedelta(days=n)) for n in range(1,29)]
    nz28  = float(np.mean(np.array(arr28)>0)) if arr28 else 0.0
    dsl=365.0
    for n,v in enumerate(arr28,1):
        if v>0: dsl=float(n); break

    # 달력 플래그(사용자 제공)
    flags = get_flags_for_date(cal_map, target_date)

    out = {
        "dow": target_date.weekday(),
        "month": target_date.month,
        "lag1": l1, "lag7": l7, "lag14": l14, "lag28": l28,
        "rm7": rm7, "rs7": rs7, "rmax7": rmax7, "rmin7": rmin7,
        "rm14": rm14, "rs14": rs14, "rmax14": rmax14, "rmin14": rmin14,
        "rm28": rm28, "rs28": rs28, "rmax28": rmax28, "rmin28": rmin28,
        "nz28": nz28, "dsl": dsl,
        # 달력 플래그
        "is_holiday": flags.get("is_holiday", 0),
        "is_before_holiday": flags.get("is_before_holiday", 0),
        "is_after_holiday": flags.get("is_after_holiday", 0),
        "is_sandwich": flags.get("is_sandwich", 0),
        "between_holidays": flags.get("between_holidays", 0),
        "is_weekend": flags.get("is_weekend", 0),
    }
    return out

FEATS = [
    "dow","month","lag1","lag7","lag14","lag28",
    "rm7","rs7","rmax7","rmin7","rm14","rs14","rmax14","rmin14",
    "rm28","rs28","rmax28","rmin28","nz28","dsl",
    # 달력 플래그
    "is_holiday","is_before_holiday","is_after_holiday",
    "is_sandwich","between_holidays","is_weekend",
]

# ===== 라우팅/ZI =====
def routing_stats_from_last28(series,last_date):
    idx=[last_date-timedelta(days=d) for d in range(27,-1,-1)]
    vals=np.array([pull_val(series,d) for d in idx],float)
    zero_ratio=float((vals==0).mean()) if vals.size else 1.0
    pos=vals[vals>0]; n_pos=int(pos.size)
    pos_mean=pos.mean() if n_pos else 0.0
    pos_std =pos.std(ddof=0) if n_pos else 0.0
    cv=pos_std/pos_mean if pos_mean>0 else np.inf
    var=pos.var(ddof=0) if n_pos else 0.0
    phi=var/pos_mean if pos_mean>1e-9 else np.inf
    p95=np.percentile(pos,95) if n_pos else 0.0
    return dict(zero_ratio=zero_ratio,n_pos=n_pos,cv=cv,phi=phi,
                p_nonzero=1-zero_ratio,pos_mean=pos_mean,pos_p95=p95)

def zi_surrogate_pred(stats,model_type):
    base=stats["p_nonzero"]*stats["pos_mean"]
    if model_type=="ZINB":
        shrink=1.0/(1.0+max(0.0,stats["phi"]-1.0))
        return max(0.0, base*(0.7+0.3*shrink))
    return max(0.0,base)

# ===== 라이브러리 로드 =====
xgb_ok=True
try:
    import xgboost as xgb
except Exception:
    xgb_ok=False

lgb_ok=True
try:
    import lightgbm as lgb
except Exception:
    lgb_ok=False

dl_ok=True
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass
    tf.random.set_seed(42)
except Exception:
    dl_ok=False

from sklearn.ensemble import GradientBoostingRegressor

# ===== 학습 데이터 구성 =====
def build_supervised_rows(train, cal_map):
    rows,ys=[],[]
    for _,g in train.groupby("영업장명_메뉴명"):
        s=normalize_series_from_group(g)
        for i,dt in enumerate(s.index):
            past=s.iloc[:i]
            if past.size<28: continue
            y=float(s.iloc[i])
            if y<=0: continue  # 양수일만
            rows.append(build_feats_from_series(past,dt,cal_map)); ys.append(y)
    X=pd.DataFrame(rows,columns=FEATS); y=np.array(ys,float)
    if X.empty: raise RuntimeError("학습 샘플 없음")
    return X,y

def build_sequence_dataset(train, win=28):
    X_seq, y_seq = [], []
    for _,g in train.groupby("영업장명_메뉴명"):
        s = normalize_series_from_group(g).astype(float)
        vals = s.values
        if len(vals) <= win: continue
        for i in range(win, len(vals)):
            xw = vals[i-win:i]
            y  = vals[i]
            X_seq.append(np.log1p(xw))
            y_seq.append(np.log1p(max(y, 0.0)))
    if not X_seq: raise RuntimeError("시퀀스 학습 샘플 없음")
    X_seq = np.array(X_seq, dtype=np.float32)[..., None]
    y_seq = np.array(y_seq, dtype=np.float32)
    return X_seq, y_seq

def time_order_split(X, y, valid_ratio=0.1):
    n = len(X)
    n_val = max(1, int(n * valid_ratio))
    idx_train = np.arange(0, n - n_val)
    idx_val   = np.arange(n - n_val, n)
    return (X.iloc[idx_train], y[idx_train]), (X.iloc[idx_val], y[idx_val])

# ===== sMAPE =====
def _smape(y_true, y_pred):
    y_true = np.array(y_true, float)
    y_pred = np.array(y_pred, float)
    y_pred = np.clip(y_pred, 1e-9, None)
    return (200.0/len(y_true)) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# ===== XGBoost =====
def train_xgb_model(train, cal_map, seeds=(42, 202, 777)):
    X, y = build_supervised_rows(train, cal_map)
    (X_tr, y_tr), (X_val, y_val) = time_order_split(X, y, VALID_RATIO)

    if xgb_ok:
        use_gpu = True
        models = []
        for sd in seeds:
            dtr  = xgb.DMatrix(X_tr,  label=np.log1p(y_tr))
            dval = xgb.DMatrix(X_val, label=np.log1p(y_val))
            params=dict(
                objective="reg:squarederror",
                eval_metric="rmse",
                tree_method="gpu_hist" if use_gpu else "hist",
                predictor="gpu_predictor" if use_gpu else "auto",
                max_bin=256,
                eta=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=8,
                reg_lambda=1.0,
                seed=sd
            )
            model = xgb.train(
                params, dtr,
                num_boost_round=NUM_ROUNDS_XGB,
                evals=[(dtr, "train"), (dval, "valid")],
                early_stopping_rounds=EARLY_STOP_XGB,
                verbose_eval=False
            )
            models.append(model)

        def _predict_with_best(model, dmatrix):
            bi = getattr(model, "best_iteration", None)
            if isinstance(bi, (int, np.integer)) and bi >= 0:
                return model.predict(dmatrix, iteration_range=(0, int(bi) + 1))
            bntl = getattr(model, "best_ntree_limit", None)
            if isinstance(bntl, (int, np.integer)) and bntl > 0:
                try:
                    return model.predict(dmatrix, iteration_range=(0, int(bntl)))
                except Exception:
                    pass
            try:
                rounds = model.num_boosted_rounds()
                return model.predict(dmatrix, iteration_range=(0, int(rounds)))
            except Exception:
                return model.predict(dmatrix)

        def predict(Xdf):
            d = xgb.DMatrix(Xdf)
            ps = [_predict_with_best(m, d) for m in models]
            ps = [np.expm1(np.clip(p, 0, None)) for p in ps]
            return np.mean(ps, axis=0)
    else:
        (X_tr, y_tr), _ = time_order_split(X, y, VALID_RATIO)
        gbr=GradientBoostingRegressor(learning_rate=0.05,n_estimators=1000,
                                      max_depth=6,subsample=0.9)
        gbr.fit(X_tr, np.log1p(y_tr))
        def predict(Xdf): return np.expm1(np.clip(gbr.predict(Xdf),0,None))
    return predict

# ===== LightGBM =====
def train_lgb_model(train, cal_map, seeds=(13, 101)):
    X, y = build_supervised_rows(train, cal_map)
    (X_tr, y_tr), (X_val, y_val) = time_order_split(X, y, VALID_RATIO)

    if lgb_ok:
        models = []
        for sd in seeds:
            dtr  = lgb.Dataset(X_tr,  label=np.log1p(y_tr), free_raw_data=False)
            dval = lgb.Dataset(X_val, label=np.log1p(y_val), free_raw_data=False)

            params = dict(
                objective="regression",
                metric="l1",
                learning_rate=0.05,
                num_leaves=31,
                feature_fraction=0.88,
                bagging_fraction=0.80,
                bagging_freq=1,
                min_data_in_leaf=40,
                lambda_l2=2.0,
                max_bin=255,
                device="gpu",
                gpu_platform_id=-1,
                gpu_device_id=0,
                seed=sd
            )

            callbacks = [
                lgb.early_stopping(EARLY_STOP_LGB, verbose=False),
                lgb.log_evaluation(period=0)
            ]

            m = lgb.train(
                params,
                dtr,
                num_boost_round=NUM_ROUNDS_LGB,
                valid_sets=[dtr, dval],
                valid_names=["train","valid"],
                callbacks=callbacks
            )
            models.append(m)

        def predict(Xdf):
            ps = [m.predict(Xdf, num_iteration=m.best_iteration) for m in models]
            p  = np.mean(ps, axis=0)
            return np.expm1(np.clip(p, 0, None))
    else:
        def predict(Xdf): return np.zeros(len(Xdf), dtype=float)

    return predict

# ===== CNN-LSTM =====
def build_sequence_dataset_only(train, win=28):
    X_seq, y_seq = [], []
    for _,g in train.groupby("영업장명_메뉴명"):
        s = normalize_series_from_group(g).astype(float)
        vals = s.values
        if len(vals) <= win: continue
        for i in range(win, len(vals)):
            xw = vals[i-win:i]
            y  = vals[i]
            X_seq.append(np.log1p(xw))
            y_seq.append(np.log1p(max(y, 0.0)))
    if not X_seq: raise RuntimeError("시퀀스 학습 샘플 없음")
    X_seq = np.array(X_seq, dtype=np.float32)[..., None]
    y_seq = np.array(y_seq, dtype=np.float32)
    return X_seq, y_seq

def train_cnn_lstm_model(train, win=35, seed=1234):
    if not dl_ok:
        def predict(seq_): return np.zeros(len(seq_), dtype=float)
        return predict
    try: tf.random.set_seed(seed)
    except Exception: pass

    X_seq, y_seq = build_sequence_dataset_only(train, win=win)

    n = len(X_seq); n_val = max(1, int(n * VALID_RATIO))
    X_tr, y_tr = X_seq[:n - n_val], y_seq[:n - n_val]
    X_val, y_val = X_seq[n - n_val:], y_seq[n - n_val:]

    ds_tr  = tf.data.Dataset.from_tensor_slices((X_tr,  y_tr)).batch(2048).prefetch(tf.data.AUTOTUNE)
    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(2048).prefetch(tf.data.AUTOTUNE)

    inputs = keras.Input(shape=(win,1))
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(inputs)
    x = layers.Dropout(DL_DROPOUT)(x)
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(DL_DROPOUT)(x)
    outputs = layers.Dense(1, dtype="float32")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(3e-4), loss="mae")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
    ]
    model.fit(ds_tr, epochs=50, validation_data=ds_val, verbose=0, callbacks=callbacks)

    def predict(seq_batch):
        if not seq_batch: return np.array([], dtype=float)
        Xb = np.array([np.log1p(np.array(s, dtype=np.float32)) for s in seq_batch])[..., None]
        pred_log = model.predict(Xb, verbose=0).reshape(-1)
        return np.expm1(np.maximum(pred_log, 0.0))
    return predict

# ===== 메인 =====
def main():
    # --- 데이터 로딩
    train_path=find_train_path()
    train=ensure_basic_cols(safe_read_csv(train_path))
    sample=None
    if os.path.exists(SAMPLE_SUB):
        sample=safe_read_csv(SAMPLE_SUB)

    train["영업일자"]=pd.to_datetime(train["영업일자"],errors="coerce")
    train["매출수량"]=pd.to_numeric(train["매출수량"],errors="coerce").fillna(0).clip(lower=0)

    # --- 달력 플래그 맵(학습 전체 기준)
    train_cal_map = build_calendar_flag_map(train)

    # --- 모델 학습
    predict_xgb = train_xgb_model(train, train_cal_map)
    predict_lgb = train_lgb_model(train, train_cal_map)
    predict_dl  = train_cnn_lstm_model(train, win=DL_WIN)

    # --- 트리 가중치 간단 탐색
    try:
        X_sup, y_sup = build_supervised_rows(train, train_cal_map)
        (_, _), (X_val, y_val) = time_order_split(X_sup, y_sup, VALID_RATIO)
        p_x = predict_xgb(X_val) if xgb_ok else np.zeros(len(X_val))
        p_l = predict_lgb(X_val) if lgb_ok else np.zeros(len(X_val))
        cand_tree = [0.30, 0.45, 0.50, 0.55, 0.60]
        best_smape, best_w_tree = 1e9, 0.55
        for wt in cand_tree:
            if xgb_ok and lgb_ok:
                p_tree = wt * p_x + (1.0 - wt) * p_l
            else:
                p_tree = p_x if xgb_ok else p_l
            s = _smape(y_val, p_tree)
            if s < best_smape:
                best_smape, best_w_tree = s, wt
        W_TREE = float(best_w_tree)
        W_DL   = 0.15
        print(f"[VAL] 선택된 앙상블 가중치(트리 전용): W_TREE={W_TREE:.2f} (SMAPE={best_smape:.4f})")
    except Exception as e:
        print("[VAL] 가중치 선택 스킵:", e)
        W_TREE, W_DL = 0.55, 0.15

    # --- 예측
    all_rows=[]
    test_files = list_test_files()
    for tf in test_files:
        tdf=ensure_basic_cols(safe_read_csv(tf))
        tdf["영업일자"]=pd.to_datetime(tdf["영업일자"],errors="coerce")
        tdf["매출수량"]=pd.to_numeric(tdf["매출수량"],errors="coerce").fillna(0).clip(lower=0)
        # 테스트 파일 내 달력 플래그(있으면 사용)
        test_cal_map = build_calendar_flag_map(tdf)

        # train 맵과 머지(테스트에 없는 날짜는 train의 기준 사용 가능)
        merged_cal_map = dict(train_cal_map)
        merged_cal_map.update(test_cal_map)

        tag=os.path.basename(tf).split("_")[1]  # TEST_XX_processed.csv → XX
        test_tag=f"TEST_{tag}"
        last_date=tdf["영업일자"].max()

        for item,g in tdf.groupby("영업장명_메뉴명"):
            s=normalize_series_from_group(g)
            stats=routing_stats_from_last28(s,last_date)

            use_zi=(stats["zero_ratio"]>=ZI_ZERO_RATIO and stats["cv"]>=ZI_CV and stats["n_pos"]<=ZI_N_POS)
            if use_zi:
                zi_type="ZIP" if stats["phi"]<=1.3 else "ZINB"
                alpha=float(np.clip(0.1+0.15*max(0.0,stats["phi"]-1.3),0.2,ALPHA_MAX))
                zi_base=zi_surrogate_pred(stats,zi_type)
            else:
                alpha,zi_base=0.0,0.0

            cap = min(stats["pos_p95"]*2.8, stats["pos_mean"]*3.5) if stats["n_pos"]>0 else 5.0
            cap = max(cap, 3.0)

            cur=s.copy()
            for k in range(1,8):
                td=last_date+timedelta(days=k)
                feats=build_feats_from_series(cur,td,merged_cal_map)
                Xdf=pd.DataFrame([feats],columns=FEATS).fillna(0)

                # 트리 앙상블
                yx = float(predict_xgb(Xdf)[0]) if xgb_ok else 0.0
                yl = float(predict_lgb(Xdf)[0]) if lgb_ok else 0.0
                if (not xgb_ok) and (not lgb_ok):
                    y_tree = yx
                else:
                    if xgb_ok and lgb_ok:
                        y_tree = W_TREE * yx + (1.0 - W_TREE) * yl
                    else:
                        y_tree = yx if xgb_ok else yl

                # DL 혼합
                seq = [pull_val(cur, td - timedelta(days=n)) for n in range(DL_WIN, 0, -1)]
                ydl = float(predict_dl([seq])[0]) if dl_ok else y_tree
                y_ens = (1.0 - W_DL) * y_tree + W_DL * ydl if dl_ok else y_tree

                # ZI 보조 + 가드레일
                yhat = (1.0 - alpha)*y_ens + alpha*zi_base
                yhat = max(1.0, min(yhat, cap))
                all_rows.append((f"{test_tag}+{k}일",item,yhat))
                cur.loc[td]=yhat

    pred=pd.DataFrame(all_rows,columns=["영업일자","영업장명_메뉴명","pred"])
    wide=pred.pivot_table(index="영업일자",columns="영업장명_메뉴명",values="pred",aggfunc="sum").fillna(1.0)

    # 제출 형식
    if sample is None:
        idx = sorted(wide.index.tolist())
        cols = sorted(wide.columns.tolist())
        out = pd.DataFrame({"영업일자": idx})
        for c in cols: out[c]=wide[c].reindex(idx).fillna(1.0).values
    else:
        out = sample.copy()
        idx=out["영업일자"].tolist(); cols=out.columns[1:]
        vals=[]
        for lbl in idx:
            if lbl in wide.index:
                sr=wide.loc[lbl]
                vals.append([sr.get(c,1.0) for c in cols])
            else:
                vals.append([1.0]*len(cols))
        df_vals=pd.DataFrame(vals,columns=cols)
        out = pd.concat([pd.DataFrame({"영업일자": idx}), df_vals], axis=1)

    # 정수 반올림 + 최종 클립
    for c in out.columns[1:]:
        out[c] = np.maximum(1, np.rint(out[c]).astype(int))

    # 저장
    save_path = OUT_FILE
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    except Exception:
        save_path = r"/mnt/data/submission_cnn_lstm_xgb_lgbm.csv"

    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved: {save_path}")

if __name__=="__main__":
    main()