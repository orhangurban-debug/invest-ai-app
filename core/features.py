# core/features.py
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    x = df.copy()

    # ✅ 1. Sütun adlarını məcburi string formatına çevir
    x.columns = [str(c).strip().lower() for c in x.columns]

    # ✅ 2. Lazımi sütunlar yoxlanılır
    required_cols = {"close", "high", "low"}
    if not required_cols.issubset(set(x.columns)):
        print("⚠️ add_indicators: sütunlar tapılmadı:", x.columns)
        return pd.DataFrame()

    # ✅ 3. Texniki indikatorlar hesablanır
    x["rsi"] = RSIIndicator(x["close"], window=14).rsi()
    x["ma_fast"] = SMAIndicator(x["close"], window=10).sma_indicator()
    x["ma_slow"] = SMAIndicator(x["close"], window=50).sma_indicator()
    x["adx"] = ADXIndicator(x["high"], x["low"], x["close"], window=14).adx()
    x["atr"] = AverageTrueRange(x["high"], x["low"], x["close"], window=14).average_true_range()

    bb = BollingerBands(x["close"], window=20, window_dev=2)
    x["bb_up"] = bb.bollinger_hband()
    x["bb_dn"] = bb.bollinger_lband()

    x["roc5"] = x["close"].pct_change(5) * 100
    return x.dropna().reset_index(drop=True)
