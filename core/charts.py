# core/charts.py
import pandas as pd
import plotly.graph_objects as go

def price_chart(df: pd.DataFrame, title: str = ""):
    """
    Sadə qiymət qrafiki:
      - Candlestick (open, high, low, close)
      - MA10 və MA50 xətləri
    DataFrame üçün gözlənti:
      - Sütunlar: open, high, low, close (+ istəyə bağlı: date)
      - Sütun adları fərqli (Open/Close/Adj Close) ola bilər — normallaşdırılır.
    """
    if df is None or df.empty:
        # Boş veriləndə boş fiqur qaytarırıq ki, UI çökmesin
        return go.Figure()

    x = df.copy()

    # --- Sütun adlarını normallaşdır ---
    x.columns = [str(c).lower().strip() for c in x.columns]

    # Adj Close ehtiyatı
    if "close" not in x.columns and "adj close" in x.columns:
        x["close"] = x["adj close"]

    # Tarixi indeksə qoy (əgər 'date' sütunu varsa)
    if "date" in x.columns:
        x["date"] = pd.to_datetime(x["date"], errors="coerce")
        x = x.set_index("date", drop=True)

    # İndeks datetime deyilsə, çevrilməyə cəhd et
    if not pd.api.types.is_datetime64_any_dtype(x.index):
        try:
            x.index = pd.to_datetime(x.index, errors="coerce")
        except Exception:
            # yenə də alınmasa — qrafik tarixsiz indekslə çəkiləcək
            pass

    # Lazımi sütunların yoxlanması
    need = {"open", "high", "low", "close"}
    if not need.issubset(set(x.columns)):
        # çatışmayan sütunlarda qrafiki çəkməmək daha təhlükəsizdir
        return go.Figure()

    # Sıralama (zaman artan)
    x = x.sort_index()

    # Hərəkətli ortalar
    x["ma10"] = x["close"].rolling(10, min_periods=1).mean()
    x["ma50"] = x["close"].rolling(50, min_periods=1).mean()

    # --- Plotly fiquru ---
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=x.index,
        open=x["open"],
        high=x["high"],
        low=x["low"],
        close=x["close"],
        name="Price"
    ))

    fig.add_trace(go.Scatter(
        x=x.index, y=x["ma10"],
        name="MA10", mode="lines"
    ))

    fig.add_trace(go.Scatter(
        x=x.index, y=x["ma50"],
        name="MA50", mode="lines"
    ))

    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False
    )

    return fig
