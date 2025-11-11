# core/predictor.py
import numpy as np
import pandas as pd
from typing import Literal, Dict, Any
from .ml_model import train_model, TrainedModel

def _last_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    # features-in hesabı ml_model-də də var; burada yalnız son sətri uyğunlaşdırırıq
    x = df.copy()
    x["rsi14"] = x["close"].pct_change().rolling(14).apply(
        lambda s: 100 - (100 / (1 + (s.clip(lower=0).mean() / (-(s.clip(upper=0).mean()) + 1e-9)))), raw=True
    )
    x["ma10"]  = x["close"].rolling(10).mean()
    x["ma50"]  = x["close"].rolling(50).mean()
    x["roc5"]  = x["close"].pct_change(5)
    x["atr14"] = (x["high"] - x["low"]).rolling(14).mean()
    x = x.dropna()
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
        'expected_return': 0.045,   # ~horizon üçün sadə təxmini R
        'recommendation': 'BUY'|'HOLD'|'SELL',
        'acc': 0.71
      }
    """
    tm: TrainedModel = train_model(df, horizon_days=horizon_days, model_type=model_type)
    lastX = _last_features(df, tm.X_cols)

    # proba
    if hasattr(tm.model, "predict_proba"):
        p_up = float(tm.model.predict_proba(lastX)[0, 1])
    else:
        pred = tm.model.predict(lastX)[0]
        p_up = 0.6 if pred == 1 else 0.4

    # sadə expected return: (p_up - (1-p_up)) * tipik_oynama
    vol = float(df["close"].pct_change().rolling(20).std().iloc[-1] or 0.01)
    expected_r = (p_up - (1 - p_up)) * vol * (horizon_days / 10)

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
