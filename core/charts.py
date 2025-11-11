# core/charts.py
import plotly.graph_objects as go
import pandas as pd

def price_chart(df: pd.DataFrame, title: str = ""):
    x = df.copy()
    # sütunlar lower-case gəlir: open, high, low, close
    x["ma10"] = x["close"].rolling(10).mean()
    x["ma50"] = x["close"].rolling(50).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x.index, open=x["open"], high=x["high"], low=x["low"], close=x["close"],
        name="Price"
    ))
    fig.add_trace(go.Scatter(x=x.index, y=x["ma10"], name="MA10", mode="lines"))
    fig.add_trace(go.Scatter(x=x.index, y=x["ma50"], name="MA50", mode="lines"))
    fig.update_layout(title=title, height=420, margin=dict(l=10, r=10, t=30, b=10))
    return fig
