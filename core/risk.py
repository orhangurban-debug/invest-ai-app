# core/risk.py
import math

def position_size(capital: float, risk_per_trade: float, entry: float, stop: float):
    risk_amt = max(capital * risk_per_trade, 0.0)          # məsələn 0.01 = 1%
    per_share = max(entry - stop, 0.0001)
    qty = math.floor(risk_amt / per_share)
    return max(qty, 0)

def make_trade_plan(price: float, atr: float, atr_mult_sl=2.0, atr_mult_tp=3.0):
    entry = round(price, 2)
    sl    = round(price - atr_mult_sl * atr, 2)
    tp    = round(price + atr_mult_tp * atr, 2)
    return entry, sl, tp
