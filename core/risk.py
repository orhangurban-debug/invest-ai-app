import math

def position_size(equity: float, entry_price: float, atr: float, per_trade_risk: float,
                  atr_mult_sl: float = 2.0, max_position_pct: float = 0.2):
    stop_dist = atr * atr_mult_sl
    if stop_dist <= 0 or entry_price <= 0:
        return 0
    risk_capital = equity * per_trade_risk
    qty = math.floor(risk_capital / stop_dist)
    max_qty = math.floor((equity * max_position_pct) / entry_price)
    return max(0, min(qty, max_qty))

def stop_take(entry: float, atr: float, sl_mult: float = 2.0, tp_mult: float = 4.0):
    sl = entry - sl_mult*atr
    tp = entry + tp_mult*atr
    return sl, tp
