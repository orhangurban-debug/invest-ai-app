# core/data.py
import yfinance as yf
import pandas as pd

def load_ohlcv(symbol: str, start: str, end: str, interval="1d") -> pd.DataFrame:
    """YFinance-dən OHLCV məlumatlarını yükləyir və sütun adlarını standartlaşdırır."""
    try:
        df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
        if df is None or df.empty:
            print(f"⚠️ Boş məlumat: {symbol}")
            return pd.DataFrame()

        # sütunları standart hala salırıq
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "close",
            "Volume": "volume"
        })
        df = df.dropna().reset_index()
        print(f"✅ {symbol} loaded: {df.shape}, columns={list(df.columns)}")
        return df

    except Exception as e:
        print(f"❌ Data yükləmə xətası ({symbol}): {e}")
        return pd.DataFrame()

def load_many(symbols, start, end, interval="1d"):
    """Bir neçə simvol üçün məlumatları ardıcıl yükləyir."""
    data = {}
    for sym in symbols:
        df = load_ohlcv(sym, start, end, interval)
        data[sym] = df
    return data
