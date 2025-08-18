# -*- coding: utf-8 -*-
"""
LG Aimers — 7일 메뉴 수요예측 (FINAL)
전략:
- Base: XGBoost (양수일만 log1p 회귀) / 미설치 시 GradientBoosting 대체
- 라우팅: 최근 28일에서
    zero_ratio ≥ 0.65 AND CV_pos ≥ 1.2 AND n_pos ≤ 6 → ZI 보조
      φ ≤ 1.3 → ZIP, φ > 1.3 → ZINB
- 블렌딩: ŷ = (1-α)·ŷ_XGB + α·ŷ_ZI, α = clip(0.1+0.15·(φ-1.3), 0.2, 0.5)
- 가드레일: 0 ≤ pred ≤ min(p95*3.0, mean*3.5)
- 출력: sample_submission과 동일 형식(영업일자 'TEST_XX+1일' ~ 'TEST_XX+7일'), 정수 반올림
"""

import os, glob
import numpy as np
import pandas as pd
from datetime import timedelta

# ===== 경로 설정 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CAND = [os.path.join(BASE_DIR, "re_train_01.csv"),
              os.path.join(BASE_DIR, "train.csv")]
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TEST_GLOB  = os.path.join(BASE_DIR, "TEST_*processed.csv")
OUT_PATH   = os.path.join(BASE_DIR, "submission.csv")

# ===== 유틸 =====
def safe_read_csv(path):
    for enc in ("utf-8-sig","utf-8","cp949","euc-kr"):
        try: return pd.read_csv(path, encoding=enc)
        except Exception: pass
    raise RuntimeError(f"CSV 로드 실패: {path}")

def find_train_path():
    for p in TRAIN_CAND:
        if os.path.exists(p): return p
    raise FileNotFoundError("훈련 데이터(re_train_01.csv/train.csv) 없음")

def ensure_basic_cols(df):
    for c in ["영업일자","영업장명_메뉴명","매출수량"]:
        if c not in df.columns: raise ValueError(f"필수 컬럼 누락: {c}")
    return df

def normalize_series_from_group(g):
    s = g.groupby(g["영업일자"].dt.normalize())["매출수량"].sum().astype(float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()

def pull_val(series, d):
    v = series.get(d, 0.0)
    if isinstance(v, pd.Series): v = v.sum()
    return float(v)

# ===== 피처 =====
def build_feats_from_series(values, target_date):
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
    return {"dow":target_date.weekday(),"month":target_date.month,
            "lag1":l1,"lag7":l7,"lag14":l14,"lag28":l28,
            "rm7":rm7,"rs7":rs7,"rmax7":rmax7,"rmin7":rmin7,
            "rm14":rm14,"rs14":rs14,"rmax14":rmax14,"rmin14":rmin14,
            "rm28":rm28,"rs28":rs28,"rmax28":rmax28,"rmin28":rmin28,
            "nz28":nz28,"dsl":dsl}

FEATS = ["dow","month","lag1","lag7","lag14","lag28",
         "rm7","rs7","rmax7","rmin7","rm14","rs14","rmax14","rmin14",
         "rm28","rs28","rmax28","rmin28","nz28","dsl"]

# ===== 라우팅 =====
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

# ===== 모델 =====
xgb_ok=True
try: import xgboost as xgb
except: xgb_ok=False
from sklearn.ensemble import GradientBoostingRegressor

def train_base_model(train):
    rows,ys=[],[]
    for _,g in train.groupby("영업장명_메뉴명"):
        s=normalize_series_from_group(g)
        for i,dt in enumerate(s.index):
            past=s.iloc[:i]
            if past.size<28: continue
            y=float(s.iloc[i])
            if y<=0: continue
            rows.append(build_feats_from_series(past,dt)); ys.append(y)
    X=pd.DataFrame(rows,columns=FEATS); y=np.array(ys,float)
    if X.empty: raise RuntimeError("학습 샘플 없음")
    if xgb_ok:
        dtr=xgb.DMatrix(X, label=np.log1p(y))
        params=dict(objective="reg:squarederror",eval_metric="rmse",tree_method="hist",
                    eta=0.05,max_depth=8,subsample=0.9,colsample_bytree=0.9,
                    min_child_weight=6,reg_lambda=2.0,seed=42)
        model=xgb.train(params,dtr,num_boost_round=1200,verbose_eval=False)
        def predict(Xdf): return np.expm1(np.clip(model.predict(xgb.DMatrix(Xdf)),0,None))
    else:
        gbr=GradientBoostingRegressor(learning_rate=0.05,n_estimators=600,
                                      max_depth=5,subsample=0.9)
        gbr.fit(X,np.log1p(y))
        def predict(Xdf): return np.expm1(np.clip(gbr.predict(Xdf),0,None))
    return predict

# ===== 메인 =====
def main():
    train_path=find_train_path()
    train=ensure_basic_cols(safe_read_csv(train_path))
    sample=safe_read_csv(SAMPLE_SUB)
    train["영업일자"]=pd.to_datetime(train["영업일자"],errors="coerce")
    train["매출수량"]=pd.to_numeric(train["매출수량"],errors="coerce").fillna(0).clip(lower=0)
    predict_base=train_base_model(train)

    all_rows=[]
    for tf in sorted(glob.glob(TEST_GLOB)):
        tdf=ensure_basic_cols(safe_read_csv(tf))
        tdf["영업일자"]=pd.to_datetime(tdf["영업일자"],errors="coerce")
        tdf["매출수량"]=pd.to_numeric(tdf["매출수량"],errors="coerce").fillna(0).clip(lower=0)
        tag=os.path.basename(tf).split("_")[1]
        test_tag=f"TEST_{tag}"; last_date=tdf["영업일자"].max()
        for item,g in tdf.groupby("영업장명_메뉴명"):
            s=normalize_series_from_group(g)
            stats=routing_stats_from_last28(s,last_date)
            use_zi=(stats["zero_ratio"]>=0.65 and stats["cv"]>=1.2 and stats["n_pos"]<=6)
            if use_zi:
                zi_type="ZIP" if stats["phi"]<=1.3 else "ZINB"
                alpha=np.clip(0.1+0.15*max(0.0,stats["phi"]-1.3),0.2,0.5)
                zi_base=zi_surrogate_pred(stats,zi_type)
            else:
                alpha,zi_base=0.0,0.0
            cap=min(stats["pos_p95"]*3.0, stats["pos_mean"]*3.5) if stats["n_pos"]>0 else 5.0
            cur=s.copy()
            for k in range(1,8):
                td=last_date+timedelta(days=k)
                feats=build_feats_from_series(cur,td)
                Xdf=pd.DataFrame([feats],columns=FEATS).fillna(0)
                yx=float(predict_base(Xdf)[0]); yz=zi_base
                yhat=(1-alpha)*yx+alpha*yz; yhat=max(0,min(yhat,cap))
                all_rows.append((f"{test_tag}+{k}일",item,yhat))
                cur.loc[td]=yhat

    pred=pd.DataFrame(all_rows,columns=["영업일자","영업장명_메뉴명","pred"])
    wide=pred.pivot_table(index="영업일자",columns="영업장명_메뉴명",values="pred",aggfunc="sum").fillna(0)

    out=sample.copy()
    idx=out["영업일자"].tolist(); cols=out.columns[1:]
    vals=[]
    for lbl in idx:
        if lbl in wide.index: sr=wide.loc[lbl]; vals.append([sr.get(c,0.0) for c in cols])
        else: vals.append([0.0]*len(cols))
    final_df=pd.DataFrame(vals,columns=cols)
    final_df=final_df.round(0).clip(lower=0).astype(int)  # ✅ pandas clip 사용
    final_df.insert(0,"영업일자",idx)
    final_df.to_csv(OUT_PATH,index=False,encoding="utf-8-sig")
    print(f"[OK] Saved: {OUT_PATH}")

if __name__=="__main__":
    main()
