# core/data.py
import time
from typing import Dict, List
import pandas as pd
import yfinance as yf

def _download_one(symbol: str, start: str, end: str, interval: str, retries: int = 4, base_sleep: float = 1.5) -> pd.DataFrame:
    last_err = None
    for i in range(retries):
        try:
            df = yf.download(symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=True, threads=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.reset_index(drop=False)
                # sütunları kiçilt və standartlaşdır
                df.columns = [str(c).lower() for c in df.columns]
                # bəzən yfinance "adj close" verir – "close" yoxdursa onu istifadə edək
                if "close" not in df and "adj close" in df:
                    df["close"] = df["adj close"]
                # yalnız lazım olan sütunların olduğuna əmin ol
                need = {"open","high","low","close"}
                if not need.issubset(set(df.columns)):
                    raise ValueError(f"Missing columns: {need - set(df.columns)}")
                return df.dropna()
            last_err = RuntimeError("Empty dataframe")
        except Exception as e:
            last_err = e
        time.sleep(base_sleep * (2**i))  # exponential backoff
    raise last_err

def load_many(symbols: List[str], start: str, end: str, interval: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        out[s] = _download_one(s, start, end, interval)
        time.sleep(0.6)  # bir az pauza – limitə düşməmək üçün
    return out
