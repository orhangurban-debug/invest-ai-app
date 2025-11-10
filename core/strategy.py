# core/strategy.py
import numpy as np
import pandas as pd

def _score_row(r):
    trend = 0
    if r["ma_fast"] > r["ma_slow"]: trend += 20
    if r["adx"] > 20: trend += 10

    momentum = 0
    if r["rsi"] < 30: momentum += 20
    elif r["rsi"] > 70: momentum -= 20
    momentum += np.clip(r.get("roc5", 0.0), -10, 10)

    vol = 10 if (r["atr"] / r["close"] < 0.03) else 0

    score = 30 + trend + momentum/2 + vol          # 0–100 aralığına sıxırıq
    return float(np.clip(score, 0, 100))

def classify_action(score: float):
    if score >= 75: return "Strong Buy"
    if score >= 60: return "Buy"
    if score >= 40: return "Hold"
    if score >= 25: return "Sell"
    return "Strong Sell"

def latest_signal(df: pd.DataFrame):
    r = df.iloc[-1]
    s = _score_row(r)
    act = classify_action(s)
    return s, act, r
