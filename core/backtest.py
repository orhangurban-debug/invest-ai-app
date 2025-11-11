# core/backtest.py
import pandas as pd
import numpy as np

def simple_signals(df: pd.DataFrame, rsi_low=30, rsi_high=70, fast_ma=10, slow_ma=50):
    x = df.copy()
    x["ma_fast"] = x["close"].rolling(fast_ma).mean()
    x["ma_slow"] = x["close"].rolling(slow_ma).mean()
    x["rsi14"] = 100 - (100 / (1 + x["close"].pct_change().rolling(14).apply(lambda s: (s.clip(lower=0).mean() / (-(s.clip(upper=0).mean()) + 1e-9)), raw=True)))
    # siqnallar (lag ilə – lookahead olmasın)
    long_cond  = (x["ma_fast"] > x["ma_slow"]) & (x["rsi14"] < rsi_high)
    short_cond = (x["ma_fast"] < x["ma_slow"]) & (x["rsi14"] > rsi_low)
    x["sig"] = 0
    x.loc[long_cond, "sig"] = 1
    x.loc[short_cond, "sig"] = -1
    x["sig"] = x["sig"].shift(1).fillna(0)
    return x

def run_backtest(df: pd.DataFrame, fee=0.0005, rsi_low=30, rsi_high=70, fast_ma=10, slow_ma=50):
    x = simple_signals(df, rsi_low, rsi_high, fast_ma, slow_ma)
    px = x["close"].copy()
    sig = x["sig"].copy()

    pos = 0  # -1, 0, +1
    equity = 1.0
    curve = []

    for i in range(1, len(px)):
        ret = 0.0
        if pos != 0:
            ret = pos * (px.iloc[i] / px.iloc[i-1] - 1.0)
            ret -= fee * abs(pos)
        # rebalans siqnalı
        if sig.iloc[i] != pos:
            pos = sig.iloc[i]
            ret -= fee * abs(pos)  # giriş/çıxış komissiyası
        equity *= (1.0 + ret)
        curve.append(equity)

    result = pd.DataFrame({"equity": [1.0] + curve}, index=df.index)
    total_ret = result["equity"].iloc[-1] - 1.0
    max_dd = (result["equity"].cummax() / result["equity"] - 1.0).max()
    return {
        "total_return": float(total_ret),
        "max_drawdown": float(max_dd),
        "equity_curve": result
    }
