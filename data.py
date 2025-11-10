import pandas as pd
import yfinance as yf

def load_ohlcv(symbol: str, start: str = None, end: str = None, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    df = df.rename(columns=str.title)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    return df
