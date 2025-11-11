# core/portfolio.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from .predictor import ai_forecast
from .risk import position_size
from .features import add_indicators

EPS = 1e-12

def _daily_returns(df: pd.DataFrame) -> pd.Series:
    x = df.copy()
    x.columns = [str(c).lower() for c in x.columns]
    ret = x["close"].pct_change().dropna()
    return ret

def calc_var_es(returns: pd.Series, alpha: float = 0.95) -> Tuple[float, float]:
    """
    Tarixi VaR və Expected Shortfall (ES).
    Qayıdışlar günlükdür. Nəticələr MƏNFİ ədəd kimi qaytarılır (risk/itki).
    """
    if len(returns) < 50:
        return (0.0, 0.0)
    q = np.quantile(returns.values, 1 - alpha)
    es = returns[returns <= q].mean() if (returns <= q).any() else q
    return float(q), float(es)

def calc_sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    """
    Sadə Sharpe (illik ~252 gün fərziyyəsi).
    """
    if len(returns) < 20:
        return 0.0
    mu = returns.mean() * 252.0
    sigma = returns.std(ddof=0) * np.sqrt(252.0)
    if sigma < EPS:
        return 0.0
    return float((mu - rf) / sigma)

def optimize_allocation(candidates: List[Dict], max_pos_pct: float = 0.25) -> Dict[str, float]:
    """
    Heüristik: müsbət expected_return varsa, score = exp_ret / (vol+eps).
    Sonra normallaşdırılır və hər simvol max_pos_pct ilə limitlənir.
    Əgər hamısı ≤0-dırsa, bərabər kiçik pay verilir.
    candidates: [{'symbol','exp_ret','vol'}]
    """
    scores = []
    for c in candidates:
        er = float(c.get("exp_ret", 0.0))
        vol = float(c.get("vol", 0.0))
        s = er / (vol + EPS) if er > 0 else 0.0
        scores.append(max(0.0, s))
    tot = sum(scores)
    weights = {}
    if tot <= EPS:
        # hamısı zəifdirsə – bərabər amma limitli
        n = max(1, len(candidates))
        w = min(max_pos_pct, 1.0 / n)
        for c in candidates:
            weights[c["symbol"]] = w
        # qalan pay boş qalır (nağd)
        return weights

    # normallaşdır, cap tətbiq et
    prelim = [s / tot for s in scores]
    # cap
    prelim = [min(p, max_pos_pct) for p in prelim]
    # yenidən normallaşdır (cap sonrası)
    s2 = sum(prelim)
    if s2 <= EPS:
        n = max(1, len(candidates))
        w = min(max_pos_pct, 1.0 / n)
        for c in candidates:
            weights[c["symbol"]] = w
        return weights

    for c, p in zip(candidates, prelim):
        weights[c["symbol"]] = p / s2
    return weights

def build_trade_plan(
    weights: Dict[str, float],
    entries: Dict[str, float],
    atrs: Dict[str, float],
    init_cash: float,
    per_trade_risk: float,
) -> Dict[str, Dict]:
    """
    Hər simvol üçün hədəf kapital = weight * init_cash.
    Qty – entry və risk (ATR) əsasında konservativ təyin olunur (position_size ilə).
    """
    plans = {}
    for sym, w in weights.items():
        target_cash = float(w) * float(init_cash)
        entry = float(entries[sym])
        atr = float(atrs.get(sym, entry * 0.02))  # ehtiyat
        # StopLoss-u təxmini: 2*ATR (position_size funksiyasında SL tələb olunur)
        sl_price = max(0.01, entry - 2.0 * atr)
        qty_by_risk = position_size(init_cash=target_cash, per_trade_risk=per_trade_risk, entry=entry, sl=sl_price)
        # kapital limitinə görə qty
        qty_by_cash = int(target_cash // max(entry, EPS))
        qty = int(max(0, min(qty_by_risk, qty_by_cash)))
        plans[sym] = {
            "weight": round(w, 4),
            "target_cash": round(target_cash, 2),
            "entry": round(entry, 4),
            "atr": round(atr, 4),
            "sl": round(sl_price, 4),
            "qty": qty
        }
    return plans

def analyze_portfolio(
    raw: Dict[str, pd.DataFrame],
    symbols: List[str],
    horizon_days: int = 5,
) -> Dict:
    """
    Hər simvol üçün exp_ret (ai_forecast), vol (20g std), returns (günlük) çıxarır.
    """
    rows = []
    rets = {}
    for sym in symbols:
        df = raw.get(sym)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        # ML proqnoz üçün indikator lazımdırsa predictor özü hesablayır; bura xam df veririk
        fx = ai_forecast(df, horizon_days=horizon_days, model_type="xgb")
        # vol (20g std, günlük)
        vol20 = float(_daily_returns(df).rolling(20).std().dropna().iloc[-1] if len(df) > 25 else 0.01)
        rows.append({
            "symbol": sym,
            "exp_ret": float(fx["expected_return"]),  # günlük horizon şkalasına uyğun
            "prob_up": float(fx["prob_up"]),
            "acc": float(fx["acc"]),
            "vol": vol20
        })
        rets[sym] = _daily_returns(df)

    return {"rows": rows, "returns": rets}
