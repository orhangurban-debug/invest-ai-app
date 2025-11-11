# core/predictor.py
import numpy as np
import pandas as pd
from .ml_model import train_model, predict_up_proba

def ai_forecast(df_feat: pd.DataFrame, horizon: int = 10, model_type: str = "xgb"):
    # treninq
    model, meta = train_model(df_feat, horizon=horizon, model_type=model_type)

    # son sətir üzrə ehtimal
    last = df_feat.iloc[-1]
    p_up = predict_up_proba(model, meta, last)  # 0..1

    # sadə gözləntilər (heuristic): ATR və vol əsasında 10 günlük gözlənən diapazon
    atr = float(df_feat["atr"].iloc[-1] if "atr" in df_feat.columns else df_feat["close"].iloc[-1]*0.02)
    price = float(df_feat["close"].iloc[-1])

    exp_up   = atr * 3.0 / price    # ~ +% diapazon (10 gün)
    exp_down = atr * 2.0 / price    # ~ -% diapazon
    exp_ret  = p_up*exp_up - (1-p_up)*exp_down

    # tövsiyə
    if p_up >= 0.62 and exp_ret > 0:
        rec = "BUY"
    elif p_up <= 0.38 and exp_ret < 0:
        rec = "SELL"
    else:
        rec = "HOLD"

    return {
        "proba_up": round(p_up, 3),
        "expected_return": round(exp_ret*100, 2),   # %
        "recommendation": rec,
        "meta": meta
    }
