# core/data.py
import time
import pandas as pd
import yfinance as yf
from typing import Dict, List


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Yfinance nəticəsini standart formata salır: open, high, low, close, volume"""
    if df is None or df.empty:
        raise ValueError("Boş DataFrame alındı")

    # MultiIndex columns (məs: ('Open', 'AAPL')) formatı üçün
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    # Əgər Adj Close varsa, onu Close kimi götür
    if "close" not in df and "adj close" in df:
        df["close"] = df["adj close"]

    # İndeksi reset et (Date varsa)
    if "date" not in df.columns:
        df = df.reset_index(drop=False)

    # Lazımi sütunları yoxla
    need = {"open", "high", "low", "close"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df.dropna(subset=list(need))


def _download_one(symbol: str, start: str, end: str, interval: str,
                  retries: int = 4, base_sleep: float = 1.5) -> pd.DataFrame:
    """Simvol üzrə məlumatı yükləyir və normallaşdırır"""
    last_err = None
    for i in range(retries):
        try:
            df = yf.download(
                symbol, start=start, end=end,
                interval=interval, progress=False,
                auto_adjust=True, threads=False
            )
            if not df.empty:
                df = _normalize_df(df)
                df["symbol"] = symbol
                return df
            last_err = RuntimeError("Empty dataframe")
        except Exception as e:
            last_err = e
        time.sleep(base_sleep * (2 ** i))  # Exponential backoff
    raise last_err


def load_many(symbols: List[str], start: str, end: str, interval: str) -> Dict[str, pd.DataFrame]:
    """Bir neçə simvol üçün məlumatı ardıcıl yükləyir"""
    out: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            df = _download_one(s, start, end, interval)
            out[s] = df
        except Exception as e:
            out[s] = pd.DataFrame()  # boş yaz, amma tətbiq çökmesin
            print(f"[WARN] {s}: {e}")
        time.sleep(0.6)
    return out
