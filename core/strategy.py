import pandas as pd
from .modeling import predict_proba

def generate_signal(df_feat: pd.DataFrame, model=None, meta=None, rsi_bounds=(30,70),
                    fast_ma=10, slow_ma=50, ml_weight=0.5):
    row = df_feat.iloc[-1:]
    rsi = float(row["rsi14"].values[0])
    ma10 = float(row["ma10"].values[0])
    ma50 = float(row["ma50"].values[0])
    price = float(row["Close"].values[0])

    tech_long = (rsi < rsi_bounds[0]) or (ma10 > ma50 and price > ma10)
    tech_short = (rsi > rsi_bounds[1]) or (ma10 < ma50 and price < ma10)

    tech_score = 0.0
    if tech_long: tech_score += 1.0
    if tech_short: tech_score -= 1.0

    ml_score = 0.0
    if model is not None and meta is not None and len(meta.get("X_cols", []))>0:
        last = row[meta["X_cols"]]
        proba_up = predict_proba(model, last.squeeze())
        ml_score = (proba_up - 0.5) * 2.0  # -1..+1

    score = (1-ml_weight)*tech_score + ml_weight*ml_score
    if score > 0.15:
        return "BUY", score
    elif score < -0.15:
        return "SELL", score
    else:
        return "HOLD", score
