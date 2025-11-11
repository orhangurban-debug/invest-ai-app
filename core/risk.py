# core/risk.py
import math

def position_size(
    capital: float = None,
    risk_per_trade: float = None,
    entry: float = None,
    stop: float = None,
    **kwargs
) -> int:
    """
    Elastik position sizing:
    - capital üçün həm capital, həm cash, həm də init_cash qəbul edir
    - stop üçün həm stop, həm də sl qəbul edir

    Nəticə: floor( risk_amount / per_share_risk )
    """

    # ---- alias-ları xəritələ
    if capital is None:
        capital = kwargs.get("cash", kwargs.get("init_cash"))
    if stop is None:
        stop = kwargs.get("sl", kwargs.get("stop_loss"))

    # Sadə yoxlamalar
    if capital is None or risk_per_trade is None or entry is None or stop is None:
        raise ValueError(
            "position_size() üçün capital(cash/init_cash), risk_per_trade, entry və stop(sl) tələb olunur."
        )

    # risk məbləği (məs: 1% = 0.01)
    risk_amt = max(float(capital) * float(risk_per_trade), 0.0)

    # bir səhm üçün risk (entry - stop); sıfıra düşməsin deyə kiçik eps
    per_share = max(float(entry) - float(stop), 0.0001)

    qty = math.floor(risk_amt / per_share)
    return max(qty, 0)


def make_trade_plan(
    price: float,
    atr: float,
    atr_mult_sl: float = 2.0,
    atr_mult_tp: float = 3.0
):
    """
    Sadə qayda: SL = price - atr_mult_sl * ATR, TP = price + atr_mult_tp * ATR
    """
    entry = round(price, 2)
    sl    = round(price - atr_mult_sl * atr, 2)
    tp    = round(price + atr_mult_tp * atr, 2)
    return entry, sl, tp


# Geri uyğunluq üçün əvvəlki ad
stop_take = make_trade_plan
