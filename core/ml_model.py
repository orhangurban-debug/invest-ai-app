# core/ml_model.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier  # opsional
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

@dataclass
class TrainedModel:
    model: Any
    X_cols: list
    acc: float

def _build_xy(df: pd.DataFrame, horizon: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    x = df.copy()

    # Sadə, sabit feature set (sənin indicators-lə uyğun)
    x["rsi14"] = x["close"].pct_change().rolling(14).apply(
        lambda s: 100 - (100 / (1 + (s.clip(lower=0).mean() / (-(s.clip(upper=0).mean()) + 1e-9)))), raw=True
    )
    x["ma10"]  = x["close"].rolling(10).mean()
    x["ma50"]  = x["close"].rolling(50).mean()
    x["roc5"]  = x["close"].pct_change(5)
    x["atr14"] = (x["high"] - x["low"]).rolling(14).mean()

    # Hədəf: horizon gün sonra > bugünkü qiymət ?
    fut = x["close"].shift(-horizon)
    y = (fut > x["close"]).astype(int)

    X = x[["close", "ma10", "ma50", "roc5", "atr14", "rsi14"]].dropna()
    y = y.loc[X.index]
    return X, y

def train_model(df: pd.DataFrame, horizon_days: int = 10, model_type: str = "xgb") -> TrainedModel:
    X, y = _build_xy(df, horizon_days)

    if len(X) < 300 or y.nunique() < 2:
        # Data çox azdırsa RF ilə kiçik model
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(X, y)
        acc = float((clf.predict(X) == y).mean())
        return TrainedModel(model=clf, X_cols=list(X.columns), acc=acc)

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
        clf = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42, n_jobs=-1)

    clf.fit(Xtr, ytr)
    acc = float(accuracy_score(yte, clf.predict(Xte)))
    return TrainedModel(model=clf, X_cols=list(X.columns), acc=acc)
