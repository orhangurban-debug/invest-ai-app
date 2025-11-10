# core/features.py
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    RSI, MA(10/50), ADX, ATR, Bollinger və ROC5 əlavə edir.
    NaN/inf və sütun adları problemlərinə qarşı dayanıqlıdır.
    """
    x = df.copy()

    # Sütun adlarını kiçilt (Close→close və s.)
    x.columns = [c.lower() for c in x.columns]

    # Lazımi sütunlar
    for col in ["close", "high", "low"]:
        if col not in x.columns:
            raise ValueError(f"'{col}' sütunu tapılmadı (mövcud: {list(x.columns)})")

    # Rəqəmsallaşdır, NaN/inf təmizlə
    x = x.replace([np.inf, -np.inf], np.nan)
    for col in ["close", "high", "low"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["close", "high", "low"])

    # Qısa tarixçə olduqda göstəricilər yarana bilməz
    if len(x) < 60:  # MA50/BB20 üçün ehtiyat hədd
        return pd.DataFrame(columns=list(df.columns) + [
            "rsi","ma_fast","ma_slow","adx","atr","bb_up","bb_dn","roc5"
        ])

    # RSI
    try:
        x["rsi"] = RSIIndicator(x["close"], window=14).rsi()
    except Exception:
        x["rsi"] = np.nan

    # Hərəkətli ortalamalar
    x["ma_fast"] = SMAIndicator(x["close"], window=10).sma_indicator()
    x["ma_slow"] = SMAIndicator(x["close"], window=50).sma_indicator()

    # ADX
    try:
        x["adx"] = ADXIndicator(x["high"], x["low"], x["close"], window=14).adx()
    except Exception:
        x["adx"] = np.nan

    # ATR
    try:
        x["atr"] = AverageTrueRange(
            high=x["high"], low=x["low"], close=x["close"], window=14
        ).average_true_range()
    except Exception:
        x["atr"] = np.nan

    # Bollinger
    try:
        bb = BollingerBands(x["close"], window=20, window_dev=2)
        x["bb_up"] = bb.bollinger_hband()
        x["bb_dn"] = bb.bollinger_lband()
    except Exception:
        x["bb_up"] = np.nan
        x["bb_dn"] = np.nan

    # ROC5
    x["roc5"] = x["close"].pct_change(5) * 100

    # Son təmizləmə
    x = x.dropna().reset_index(drop=True)
    return x
