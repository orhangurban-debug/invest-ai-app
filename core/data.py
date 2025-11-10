# core/data.py
from yahooquery import Ticker
import pandas as pd

def load_ohlcv(symbol: str, start: str, end: str, interval="1d") -> pd.DataFrame:
    try:
        data = Ticker(symbol).history(start=start, end=end, interval=interval)
        if data.empty:
            return pd.DataFrame()

        # yahooquery çox vaxt MultiIndex qaytarır
        if isinstance(data.index, pd.MultiIndex):
            data = data.droplevel(0)

        df = data.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adjclose": "close",
            "volume": "volume"
        }).dropna().reset_index()

        print(f"✅ {symbol} loaded: {df.shape}, columns={list(df.columns)}")
        return df
    except Exception as e:
        print(f"⚠️ YahooQuery error ({symbol}): {e}")
        return pd.DataFrame()
