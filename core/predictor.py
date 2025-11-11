# core/predictor.py
import numpy as np
import pandas as pd
from typing import Literal, Dict, Any
from .ml_model import train_model, TrainedModel


def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    y = df.copy()
    y.columns = [str(c).lower() for c in y.columns]
    required = {"open", "high", "low", "close"}
    missing = required - set(y.columns)
    if missing:
        raise ValueError(f"OHLC sütunları çatışmır: {missing}")
    y = y.sort_index()
    y = y[~y.index.duplicated(keep="last")]
    for c in ["open", "high", "low", "close"]:
        y[c] = pd.to_numeric(y[c], errors="coerce")
    return y


def _last_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    ml_model._build_xy ilə eyni feature-lərin son sətirini qurur.
    """
    x = _ensure_ohlc(df).copy()
    # eyni formulalar
    x["ma10"] = x["close"].rolling(10).mean()
    x["ma50"] = x["close"].rolling(50).mean()
    x["roc5"] = x["close"].pct_change(5)
    x["atr14"] = (x["high"] - x["low"]).rolling(14).mean()

    # RSI (ml_model-dəki ilə uyğundur)
    delta = x["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-12)
    x["rsi14"] = 100 - (100 / (1 + rs))

    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        raise ValueError("Son xüsusiyyətləri hesablamaq üçün kifayət qədər sətir yoxdur.")
    # Modelin gözlədiyi sütun ardıcıllığı
    return x[cols].iloc[[-1]]


def ai_forecast(
    df: pd.DataFrame,
    horizon_days: int = 10,
    model_type: Literal["xgb", "rf"] = "xgb"
) -> Dict[str, Any]:
    """
    Çıxış:
      {
        'prob_up': 0.63,
        'expected_return': 0.045,   # ~horizon üçün təxmini R
        'recommendation': 'BUY'|'HOLD'|'SELL',
        'acc': 0.71
      }
    """
    tm: TrainedModel = train_model(df, horizon_days=horizon_days, model_type=model_type)
    lastX = _last_features(df, tm.X_cols)

    # Proqnoz ehtimalı
    if hasattr(tm.model, "predict_proba"):
        p_up = float(tm.model.predict_proba(lastX)[0, 1])
    else:
        pred = tm.model.predict(lastX)[0]
        p_up = 0.6 if pred == 1 else 0.4

    # Sadə expected return: (p_up - (1-p_up)) * tipik vol * (horizon/10)
    df2 = _ensure_ohlc(df)
    vol_20 = float(df2["close"].pct_change().rolling(20).std().iloc[-1] or 0.01)
    expected_r = (p_up - (1 - p_up)) * vol_20 * (horizon_days / 10)

    if p_up >= 0.6 and expected_r > 0:
        reco = "BUY"
    elif p_up <= 0.4 and expected_r < 0:
        reco = "SELL"
    else:
        reco = "HOLD"

    return {
        "prob_up": round(p_up, 4),
        "expected_return": round(expected_r, 4),
        "recommendation": reco,
        "acc": round(tm.acc, 4),
    }
