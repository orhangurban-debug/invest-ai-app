# core/strategy.py
import pandas as pd
from typing import Tuple, Dict, Any

try:
    # optional ML dəstəyi – yoxdursa, problem deyil
    from .modeling import predict_proba
except Exception:  # modeling olmayanda ML hissəsini atlayacağıq
    predict_proba = None  # type: ignore


def _get_first(row: pd.Series, *names):
    """row-dan ilk mövcud (və NaN olmayan) sütunu götürür."""
    for n in names:
        if n in row and pd.notna(row[n]):
            return float(row[n])
    return None


def generate_signal(
    df_feat: pd.DataFrame,
    model=None,
    meta: Dict[str, Any] = None,
    rsi_bounds=(30, 70),
    fast_ma=10,
    slow_ma=50,
    ml_weight=0.5,
) -> Tuple[str, float]:
    """
    (action, score) qaytarır.
    score ∈ [-1.0, +1.0]
    """
    if df_feat is None or len(df_feat) == 0:
        return "HOLD", 0.0

    x = df_feat.copy()
    x.columns = [c.lower().strip() for c in x.columns]
    row = x.iloc[-1]

    # Sütun adlarını elastik götürürük (səndə bəzən rsi14/ma10/ma50 də olub)
    rsi    = _get_first(row, "rsi", "rsi14")
    ma_f   = _get_first(row, "ma_fast", "ma10", "sma_fast", "sma10", str(fast_ma))
    ma_s   = _get_first(row, "ma_slow", "ma50", "sma_slow", "sma50", str(slow_ma))
    price  = _get_first(row, "close", "Close")

    # Minimum tələb olunan dəyərlər yoxdursa – HOLD
    if any(v is None for v in [rsi, ma_f, ma_s, price]):
        return "HOLD", 0.0

    # Texniki siqnal
    tech_long  = (rsi < rsi_bounds[0]) or (ma_f > ma_s and price > ma_f)
    tech_short = (rsi > rsi_bounds[1]) or (ma_f < ma_s and price < ma_f)

    tech_score = 0.0
    if tech_long:
        tech_score += 1.0
    if tech_short:
        tech_score -= 1.0

    # ML skoru (opsional)
    ml_score = 0.0
    if model is not None and meta and meta.get("X_cols") and predict_proba:
        try:
            lastX = row[[c.lower() for c in meta["X_cols"]]]
            proba_up = float(predict_proba(model, lastX))
            ml_score = (proba_up - 0.5) * 2.0  # -> [-1, +1]
        except Exception:
            ml_score = 0.0

    # Birləşik skor
    score = (1 - ml_weight) * tech_score + ml_weight * ml_score  # [-1, +1]

    if score > 0.15:
        action = "BUY"
    elif score < -0.15:
        action = "SELL"
    else:
        action = "HOLD"

    return action, float(score)


def latest_signal(
    df_feat: pd.DataFrame,
    **kwargs
) -> Tuple[float, str, Dict[str, Any]]:
    """
    app.py bu imzanı gözləyir: (score_0_100, action, last_row_dict)
    """
    if df_feat is None or len(df_feat) == 0:
        return 0.0, "HOLD", {}

    action, score = generate_signal(df_feat, **kwargs)  # [-1, +1]
    score_0_100 = (score + 1.0) * 50.0                  # -> [0, 100]
    last_dict = df_feat.iloc[-1].to_dict()
    return float(score_0_100), action, last_dict
