# core/trade_log.py
import pandas as pd
from pathlib import Path
from datetime import datetime

LOG = Path("trade_log.csv")

def append_trade(row: dict):
    row = {"time": datetime.utcnow().isoformat(timespec="seconds"), **row}
    if LOG.exists():
        df = pd.read_csv(LOG)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(LOG, index=False)

def read_log() -> pd.DataFrame:
    if LOG.exists():
        return pd.read_csv(LOG)
    return pd.DataFrame(columns=["time","symbol","action","entry","sl","tp","qty","score","note"])
