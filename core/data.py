# core/data.py
import yfinance as yf
import pandas as pd

def load_ohlc(symbol: str, start: str, end: str, interval: str="1d") -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    return df.dropna().copy()

def load_many(symbols, start, end, interval="1d"):
    data = {}
    for s in symbols:
        s = s.strip().upper()
        if not s:
            continue
        df = load_ohlc(s, start, end, interval)
        if not df.empty:
            data[s] = df
    return data
