import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def build_labels(df: pd.DataFrame, horizon_days: int = 5) -> pd.Series:
    future = df["Close"].shift(-horizon_days)
    label = (future > df["Close"]).astype(int)  # 1 = yüksələcək
    return label

def train_model(df_feat: pd.DataFrame, horizon_days: int = 5, test_size: float = 0.2):
    features = df_feat.drop(columns=["Open","High","Low","Close","Volume"])
    y = build_labels(df_feat, horizon_days)
    X = features.iloc[:-horizon_days]
    y = y.iloc[:-horizon_days]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    model = RandomForestClassifier(n_estimators=400, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = (model.predict_proba(X_test)[:,1] > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    meta = {"acc": float(acc), "X_cols": list(X.columns)}
    return model, meta

def predict_proba(model, last_row: pd.Series):
    return float(model.predict_proba(last_row.values.reshape(1, -1))[0,1])
