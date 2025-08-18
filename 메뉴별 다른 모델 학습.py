# -*- coding: utf-8 -*-
"""
Menu-wise forecasting with rule-based model routing
- Train: re_train_01.csv
- Test : TEST_00_processed.csv ~ TEST_09_processed.csv (각 28일)
- Template: sample_submission.csv
- Optional: menu_model_recommendation.csv (없으면 자동 생성)

모델 라우팅(추천 라벨):
- "LightGBM / CatBoost"      -> 요일평균 + 선형추세 앙상블
- "ZIP" / "ZINB" / "Hurdle"  -> 허들(0/양수 두 단계) 예측
- "Bayesian Hierarchical"    -> 허들 예측
- "ZINB + XGB Ensemble"      -> 허들 + (요일/추세 앙상블) 평균

출력:
- submission.csv (sample_submission.csv와 동일한 틀/순서, 정수, 0 미허용 → 최소 1)

주의:
- 평가 규정 상 테스트 28일 외 lookback 불가 → 각 TEST 블록 독립 예측
- 외부 데이터 미사용
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta

# =========================
# 경로 설정 (필요 시 수정)
# =========================
DATA_DIR = r"C:\Users\user\Desktop\LG AIMERS\모델적용\7번시도(메뉴별 다른모델학습)"  # 로컬 실행 시 여러분의 경로로 교체
TRAIN_PATH = os.path.join(DATA_DIR, "re_train_01.csv")
SUB_PATH   = os.path.join(DATA_DIR, "sample_submission.csv")
OUT_PATH   = os.path.join(DATA_DIR, "submission.csv")
MENU_REC_PATH = os.path.join(DATA_DIR, "menu_model_recommendation.csv")
TEST_PATHS = [os.path.join(DATA_DIR, f"TEST_{i:02d}_processed.csv") for i in range(10)]


# =========================
# 유틸: 안전한 CSV 로더
# =========================
def read_csv_smart(path, encodings=("utf-8-sig", "cp949", "euc-kr", "utf-8")):
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    # 마지막 시도로 인코딩 미지정
    try:
        return pd.read_csv(path)
    except Exception:
        raise last_err or RuntimeError(f"Failed to read {path}")


# =========================
# 추천 라벨 준비 (없으면 생성)
# =========================
def build_menu_recommendations(train: pd.DataFrame) -> pd.DataFrame:
    stats = (
        train.groupby("영업장명_메뉴명")["매출수량"]
        .agg(
            mean_sales="mean",
            std_sales="std",
            zero_ratio=lambda x: (x == 0).mean()
        )
        .reset_index()
    )
    stats["cv"] = stats["std_sales"] / stats["mean_sales"].replace(0, 1)

    def recommend(row):
        zero_ratio = row["zero_ratio"]
        cv = row["cv"]
        mean_sales = row["mean_sales"]
        if zero_ratio < 0.3 and cv < 1:
            return "LightGBM / CatBoost"
        elif zero_ratio < 0.3 and cv >= 1:
            return "ZINB"
        elif zero_ratio >= 0.3 and cv < 1:
            return "ZIP"
        elif zero_ratio >= 0.3 and cv >= 1 and mean_sales < 2:
            return "Bayesian Hierarchical / Hurdle"
        else:
            return "ZINB + XGB Ensemble"

    stats["추천모델"] = stats.apply(recommend, axis=1)
    return stats[["영업장명_메뉴명", "추천모델", "mean_sales", "std_sales", "zero_ratio", "cv"]]


# =========================
# 라우팅 라벨 선택
# =========================
def choose_label(menu, menu_rec: pd.DataFrame, train: pd.DataFrame) -> str:
    row = menu_rec.loc[menu_rec["영업장명_메뉴명"] == menu]
    if len(row):
        return row.iloc[0]["추천모델"]
    # fallback: train으로 간이 추정
    g = train.loc[train["영업장명_메뉴명"] == menu, "매출수량"]
    if len(g) == 0:
        return "ZINB + XGB Ensemble"
    zero_ratio = (g == 0).mean()
    mean_val = g.mean()
    cv = g.std() / (mean_val if mean_val != 0 else 1)
    if zero_ratio < 0.3 and cv < 1:
        return "LightGBM / CatBoost"
    if zero_ratio < 0.3 and cv >= 1:
        return "ZINB"
    if zero_ratio >= 0.3 and cv < 1:
        return "ZIP"
    if zero_ratio >= 0.3 and cv >= 1 and mean_val < 2:
        return "Bayesian Hierarchical / Hurdle"
    return "ZINB + XGB Ensemble"


# =========================
# 간단 예측 컴포넌트
# =========================
def weekday_cycle_preds(df28, future_dates):
    # df28: ['영업일자','매출수량','요일']  최근 28일
    wk = df28.copy()
    wk["영업일자"] = pd.to_datetime(wk["영업일자"])
    weekday_means = wk.groupby("요일")["매출수량"].mean()
    weekday_nonzero_means = wk.loc[wk["매출수량"] > 0].groupby("요일")["매출수량"].mean()
    overall = wk["매출수량"].mean()

    smoothed = {}
    for d in range(7):
        m1 = weekday_means.get(d, np.nan)
        m2 = weekday_nonzero_means.get(d, np.nan)
        vals = [v for v in [m1, m2, overall] if pd.notna(v)]
        smoothed[d] = float(np.mean(vals)) if vals else 0.0

    future_wd = [pd.to_datetime(d).weekday() for d in future_dates]
    base = overall if not np.isnan(overall) else 0.0
    return np.array([smoothed.get(w, base) for w in future_wd], dtype=float)


def trend_adjustment(df28, horizon_len=7):
    # 선형추세 외삽
    y = df28["매출수량"].values.astype(float)
    x = np.arange(len(y))
    if len(y) < 2 or np.allclose(y, y[0]):
        slope = 0.0
        intercept = float(np.mean(y)) if len(y) else 0.0
    else:
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    x_future = np.arange(len(y), len(y) + horizon_len)
    pred = intercept + slope * x_future
    return np.maximum(pred, 0.0)


def hurdle_predict(df28, future_dates, global_nonzero_mean: float):
    # 허들: (1) 비제로 확률 * (2) 양수 조건 기대값
    wk = df28.copy()
    wk["영업일자"] = pd.to_datetime(wk["영업일자"])
    # (1) 요일별 비제로 확률
    p_nonzero_map = wk.groupby("요일")["매출수량"].apply(lambda s: (s > 0).mean()).to_dict()
    # (2) 요일별 양수 중앙값(안정)
    pos = wk.loc[wk["매출수량"] > 0, "매출수량"]
    if len(pos):
        default_mu = float(pos.median())
    else:
        default_mu = float(global_nonzero_mean) if not np.isnan(global_nonzero_mean) else float(wk["매출수량"].mean())

    pos_means = wk.loc[wk["매출수량"] > 0].groupby("요일")["매출수량"].median().to_dict()
    # 스무딩
    base_p = (wk["매출수량"] > 0).mean()
    for d in range(7):
        if d not in pos_means or np.isnan(pos_means[d]):
            pos_means[d] = default_mu
        if d not in p_nonzero_map or np.isnan(p_nonzero_map[d]):
            p_nonzero_map[d] = base_p

    future_wd = [pd.to_datetime(d).weekday() for d in future_dates]
    preds = []
    for w in future_wd:
        p = p_nonzero_map.get(w, base_p)
        mu = pos_means.get(w, default_mu)
        preds.append(p * mu)
    return np.array(preds, dtype=float)


def ensemble_mean(a, b):
    return 0.5 * (np.asarray(a, float) + np.asarray(b, float))


# =========================
# 메인
# =========================
def main():
    # Load data
    print("[INFO] Load train:", TRAIN_PATH)
    train = read_csv_smart(TRAIN_PATH)

    print("[INFO] Prepare menu recommendations")
    if os.path.exists(MENU_REC_PATH):
        menu_rec = read_csv_smart(MENU_REC_PATH)
    else:
        menu_rec = build_menu_recommendations(train)
        menu_rec.to_csv(MENU_REC_PATH, index=False, encoding="utf-8-sig")
        print(f"[OK] menu recommendations saved → {MENU_REC_PATH}")

    print("[INFO] Load submission template:", SUB_PATH)
    # 템플릿은 보통 cp949/euc-kr
    try:
        sub = pd.read_csv(SUB_PATH, encoding="cp949")
    except Exception:
        sub = read_csv_smart(SUB_PATH)

    # 전역 양수 평균(희소 메뉴 스무딩용)
    global_nonzero_mean = train.loc[train["매출수량"] > 0, "매출수량"].mean()
    if np.isnan(global_nonzero_mean):
        global_nonzero_mean = train["매출수량"].mean()

    # 10개 TEST 블록 처리
    all_blocks = []
    for i, tp in enumerate(TEST_PATHS):
        print(f"[INFO] Load TEST block {i:02d}:", tp)
        df = read_csv_smart(tp)
        # 날짜 처리
        df["영업일자"] = pd.to_datetime(df["영업일자"])
        last_date = df["영업일자"].max()
        future_dates = [last_date + timedelta(days=h) for h in range(1, 8)]

        menus = df["영업장명_메뉴명"].unique().tolist()
        # 7 × M 블록
        idx = [f"TEST_{i:02d}+{h}일" for h in range(1, 8)]
        block = pd.DataFrame(index=idx, columns=menus, dtype=float)

        for menu in menus:
            hist = df.loc[df["영업장명_메뉴명"] == menu].sort_values("영업일자")
            label = choose_label(menu, menu_rec, train)

            # 컴포넌트 예측
            wk_pred     = weekday_cycle_preds(hist[["영업일자", "매출수량", "요일"]], future_dates)
            trend_pred  = trend_adjustment(hist[["영업일자", "매출수량"]], horizon_len=7)
            hurdle_pred = hurdle_predict(hist[["영업일자", "매출수량", "요일"]], future_dates, global_nonzero_mean)

            # 라우팅
            if ("LightGBM" in label) or ("CatBoost" in label):
                pred = np.maximum(ensemble_mean(wk_pred, trend_pred), 0.0)
            elif ("ZINB" in label and "XGB" not in label) or ("ZIP" in label) or ("Hurdle" in label) or ("Bayesian" in label):
                pred = np.maximum(hurdle_pred, 0.0)
            elif "Ensemble" in label:
                pred = np.maximum(ensemble_mean(hurdle_pred, ensemble_mean(wk_pred, trend_pred)), 0.0)
            else:
                pred = np.maximum(ensemble_mean(wk_pred, trend_pred), 0.0)

            block.loc[:, menu] = pred

        block.insert(0, "영업일자", block.index)
        all_blocks.append(block)

    # 70행 결합
    pred_long = pd.concat(all_blocks, axis=0, ignore_index=True)

    # 템플릿 정렬/치환
    sub_out = sub.copy()
    # 예측치 주입
    pred_indexed = pred_long.set_index("영업일자")
    for col in sub_out.columns[1:]:
        if col in pred_indexed.columns:
            sub_out[col] = pred_indexed.reindex(sub_out["영업일자"])[col].values
        else:
            sub_out[col] = 0.0  # unseen menu fallback

    # 후처리: NaN→0, 음수→0, 반올림→정수, 그리고 0 금지 → 최소 1로 클램프
    for col in sub_out.columns[1:]:
        v = sub_out[col].to_numpy(dtype=float)
        v = np.nan_to_num(v, nan=0.0, posinf=None, neginf=None)
        v = np.clip(v, 0, None)
        v = np.rint(v).astype(int)
        # ★ 0 금지 규칙: 최소 1
        v[v <= 0] = 1
        sub_out[col] = v

    # 저장 (한글 컬럼 유지 위해 cp949 권장)
    try:
        sub_out.to_csv(OUT_PATH, index=False, encoding="cp949")
    except Exception:
        sub_out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"[OK] submission written → {OUT_PATH}  shape={sub_out.shape}")


if __name__ == "__main__":
    main()
