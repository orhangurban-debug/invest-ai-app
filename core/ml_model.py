# core/ml_model.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# XGBoost opsionaldır; yoxdursa RF istifadə olunacaq
try:
    from xgboost import XGBClassifier  # optional
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


@dataclass
class TrainedModel:
    model: Any
    X_cols: list
    acc: float


def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gələn DataFrame-də sütun adlarını kiçik hərflə birləşdir və tələb olunan
    'open, high, low, close' sütunlarının mövcudluğunu yoxla.
    """
    x = df.copy()
    x.columns = [str(c).lower() for c in x.columns]
    required = {"open", "high", "low", "close"}
    missing = required - set(x.columns)
    if missing:
        raise ValueError(f"OHLC sütunları çatışmır: {missing}")
    # tarixə görə sıralama və dublikatların təmizlənməsi
    x = x.sort_index()
    x = x[~x.index.duplicated(keep="last")]
    # ədədi tiplərə çevir (əmin olmaq üçün)
    for c in ["open", "high", "low", "close"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x


def _safe_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Sadə RSI hesabı; NaN/inf problemlərinə dözümlü.
    (Momentum kitabxanasından asılılıq yaratmamaq üçün yüngül implementasiya)
    """
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window).mean()
    roll_down = down.rolling(window).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _build_xy(df: pd.DataFrame, horizon: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Eyni feature set həm train, həm də son sətir proqnozu üçün istifadə olunur.
    """
    x = _ensure_ohlc(df).copy()

    # Features
    x["ma10"] = x["close"].rolling(10).mean()
    x["ma50"] = x["close"].rolling(50).mean()
    x["roc5"] = x["close"].pct_change(5)
    x["atr14"] = (x["high"] - x["low"]).rolling(14).mean()
    x["rsi14"] = _safe_rsi(x["close"], 14)

    # Hədəf: horizon gün sonra qiymət artımı?
    fut = x["close"].shift(-horizon)
    y = (fut > x["close"]).astype(int)

    # Təmizləmə
    feats = ["close", "ma10", "ma50", "roc5", "atr14", "rsi14"]
    X = x[feats].replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    return X, y


def train_model(
    df: pd.DataFrame,
    horizon_days: int = 10,
    model_type: str = "xgb"
) -> TrainedModel:
    """
    df       : OHLC sütunları olan DataFrame (index: tarixi sıra)
    horizon  : proqnoz üfüqü (gün)
    model_type: 'xgb' (mövcuddursa) və ya 'rf'
    """
    X, y = _build_xy(df, horizon_days)

    # Data yoxlaması
    if len(X) < 120 or y.nunique() < 2:
        # çox az data olduqda – yalnız RF və bütün set üzərində fit
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1
        )
        clf.fit(X, y)
        acc = float((clf.predict(X) == y).mean())
        return TrainedModel(model=clf, X_cols=list(X.columns), acc=acc)

    # Time-series split (shuffle=False)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)

    if model_type == "xgb" and _HAS_XGB:
        clf = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1
        )

    clf.fit(Xtr, ytr)

    # Əgər test set çox kiçikdirsə, ehtiyatən acc hesabını sabit saxla
    if len(Xte) >= 10:
        acc = float(accuracy_score(yte, clf.predict(Xte)))
    else:
        acc = float((clf.predict(X) == y).mean())

    return TrainedModel(model=clf, X_cols=list(X.columns), acc=acc)
