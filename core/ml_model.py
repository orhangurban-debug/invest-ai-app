# core/ml_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# df: sütunları lower-case: open, high, low, close, volume + sizin add_indicators yaratdıqları
def make_dataset(df: pd.DataFrame, horizon: int = 10):
    x = df.copy()
    # hədəf: gələcək n günlük faiz gəlir (close_{t+h} / close_t - 1)
    x["fwd_ret"] = x["close"].shift(-horizon) / x["close"] - 1.0
    # binar: yuxarı (1) / aşağı (0)
    x["label"] = (x["fwd_ret"] > 0).astype(int)

    # feature-lar: bütün rəqəmsal sütunlar
    feats = x.select_dtypes(include=[np.number]).drop(columns=["fwd_ret", "label"]).copy()
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
    y = x["label"].copy()

    # son horizon satır proqnoz üçün dəyərsizdir → atırıq
    valid = ~y.isna()
    feats, y = feats[valid], y[valid]
    return feats, y

def train_model(df_feat: pd.DataFrame, horizon: int = 10, model_type: str = "xgb"):
    X, y = make_dataset(df_feat, horizon=horizon)
    if len(X) < 200:
        raise ValueError("Model üçün kifayət qədər sətir yoxdur.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    if model_type == "xgb" and HAS_XGB:
        model = XGBClassifier(
            max_depth=4, n_estimators=400, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, eval_metric="logloss", n_jobs=2
        )
    else:
        model = RandomForestClassifier(n_estimators=400, max_depth=6, n_jobs=2)

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    meta = {"auc": auc, "X_cols": list(X.columns), "horizon": horizon, "model_type": model_type}
    return model, meta

def predict_up_proba(model, meta, row: pd.Series) -> float:
    arr = row[meta["X_cols"]].values.reshape(1, -1)
    return float(model.predict_proba(arr)[:, 1][0])
