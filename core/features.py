# core/features.py
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands

def add_indicators(df) -> pd.DataFrame:
    """
    RSI, MA, ADX, ATR, Bollinger və ROC5 əlavə edir.
    DataFrame yoxlaması və NaN/inf təmizləməsi əlavə olunub.
    """
    # ✅ 1. Əgər DataFrame deyilsə, boş DataFrame qaytar
    if not isinstance(df, pd.DataFrame):
        print("⚠️ Xəbərdarlıq: add_indicators() səhv tipdə input aldı:", type(df))
        return pd.DataFrame()

    x = df.copy()

    # ✅ 2. Əgər DataFrame boşdursa
    if x.empty or len(x.columns) == 0:
        print("⚠️ Xəbərdarlıq: add_indicators() boş DataFrame aldı")
        return pd.DataFrame()

    # ✅ 3. Sütun adlarını kiçilt
    x.columns = [str(c).lower().strip() for c in x.columns]

    # Lazımi sütunlar yoxdursa
    for col in ["close", "high", "low"]:
        if col not in x.columns:
            raise ValueError(f"'{col}' sütunu tapılmadı (mövcud: {list(x.columns)})")

    # NaN və inf təmizlənməsi
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=["close", "high", "low"])

    if len(x) < 60:
        return pd.DataFrame(columns=list(x.columns) + [
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

    # Bollinger Bands
    try:
        bb = BollingerBands(x["close"], window=20, window_dev=2)
        x["bb_up"] = bb.bollinger_hband()
        x["bb_dn"] = bb.bollinger_lband()
    except Exception:
        x["bb_up"] = np.nan
        x["bb_dn"] = np.nan

    # ROC5
    x["roc5"] = x["close"].pct_change(5) * 100

    return x.dropna().reset_index(drop=True)
