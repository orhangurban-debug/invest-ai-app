import numpy as np
import pandas as pd

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1 * delta.clip(upper=0.0)
    ma_up = up.ewm(com=period-1, adjust=False).mean()
    ma_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def make_tech_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi14"] = rsi(out["Close"], 14)
    out["ma10"] = moving_average(out["Close"], 10)
    out["ma50"] = moving_average(out["Close"], 50)
    out["atr14"] = atr(out, 14)
    out["ret1"] = out["Close"].pct_change()
    out["ret5"] = out["Close"].pct_change(5)
    out["vol_chg"] = out["Volume"].pct_change()
    out.dropna(inplace=True)
    return out
