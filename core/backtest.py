# core/backtest.py
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator

def simple_signals(df: pd.DataFrame, rsi_low=30, rsi_high=70, fast_ma=10, slow_ma=50) -> pd.DataFrame:
    """
    Sadə texniki qayda:
      - MA(fast) və MA(slow)
      - RSI(14)
      - Siqnal: +1 (long), -1 (short), 0 (flat)  -- 1 bar lag ilə (lookahead yoxdur).
    """
    x = df.copy()
    # sütunlar lower-case olmalıdır: open, high, low, close
    x.columns = [str(c).lower() for c in x.columns]

    # Hərəkətli ortalar
    x["ma_fast"] = x["close"].rolling(int(fast_ma), min_periods=int(fast_ma)).mean()
    x["ma_slow"] = x["close"].rolling(int(slow_ma), min_periods=int(slow_ma)).mean()

    # Standart RSI(14)
    x["rsi14"] = RSIIndicator(close=x["close"], window=14).rsi()

    # NaN-ları təmizlə (hesablamalardan sonra)
    x = x.dropna(subset=["close", "ma_fast", "ma_slow", "rsi14"]).copy()

    # Siqnal şərtləri
    long_cond  = (x["ma_fast"] > x["ma_slow"]) & (x["rsi14"] < rsi_high)
    short_cond = (x["ma_fast"] < x["ma_slow"]) & (x["rsi14"] > rsi_low)

    x["sig"] = 0
    x.loc[long_cond,  "sig"] = 1
    x.loc[short_cond, "sig"] = -1

    # 1 bar gecikmə – gələcəyi “görməmək” üçün
    x["sig"] = x["sig"].shift(1).fillna(0).astype(int)

    return x


def run_backtest(df: pd.DataFrame, fee=0.0005, rsi_low=30, rsi_high=70, fast_ma=10, slow_ma=50):
    """
    Sadə portfel əyrisi:
      - Tam kapitalla -1/0/+1 mövqe (heç bir qaldıraq/levərac yoxdur)
      - Gündəlik (barlıq) gəlir = pozisiya * (P_t / P_{t-1} - 1)
      - Komissiya YALNIZ siqnal dəyişikliyində tutulur (giriş/çıxış): equity *= (1 - fee * |yeni_pozisiya|)
    """
    if df is None or df.empty:
        return {"total_return": 0.0, "max_drawdown": 0.0, "equity_curve": pd.DataFrame(index=df.index if df is not None else None)}

    x = simple_signals(df, rsi_low=rsi_low, rsi_high=rsi_high, fast_ma=fast_ma, slow_ma=slow_ma)

    # Hesablamaya daxil olacaq seriyalar
    px  = x["close"]
    sig = x["sig"]

    if len(px) < 2:
        # kifayət qədər data yoxdursa
        result = pd.DataFrame({"equity": [1.0]}, index=x.index[:1])
        return {"total_return": 0.0, "max_drawdown": 0.0, "equity_curve": result}

    pos = 0            # -1, 0, +1
    equity = 1.0
    curve = [equity]   # başlanğıc 1.0
    last_price = px.iloc[0]

    for i in range(1, len(px)):
        price = px.iloc[i]
        ret = 0.0

        # Mövcud mövqeyin gündəlik gəliri (holding return)
        if pos != 0 and last_price > 0:
            ret = pos * (price / last_price - 1.0)

        # Siqnal dəyişikliyi → yeni pozisiya və komissiya
        desired = int(sig.iloc[i])
        if desired != pos:
            pos = desired
            # giriş/çıxışda komissiya – alınan mövqe ölçüsü qədər
            if pos != 0:
                ret -= fee * abs(pos)   # 1 dəfəlik komissiya (giriş)
            else:
                ret -= fee * 1.0        # çıxışda da 1 vahidlik komissiya kimi qəbul

        equity *= (1.0 + ret)
        curve.append(equity)
        last_price = price

    equity_curve = pd.DataFrame({"equity": curve}, index=x.index[:len(curve)])
    total_ret = float(equity_curve["equity"].iloc[-1] - 1.0)

    # Max Drawdown
    running_max = equity_curve["equity"].cummax()
    dd = (running_max - equity_curve["equity"]) / running_max
    max_dd = float(dd.max()) if len(dd) else 0.0

    return {
        "total_return": total_ret,
        "max_drawdown": max_dd,
        "equity_curve": equity_curve
    }
