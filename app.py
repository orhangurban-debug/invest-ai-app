import os, time, json, datetime

def log_action(kind, payload: dict):
    import csv, os, datetime
    os.makedirs("logs", exist_ok=True)
    with open("logs/actions.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([datetime.datetime.utcnow().isoformat(), kind, json.dumps(payload, ensure_ascii=False)])

import pandas as pd
import streamlit as st
from datetime import date

from core.data import load_ohlcv
from core.features import make_tech_features
from core.modeling import train_model
from core.strategy import generate_signal
from core.risk import position_size, stop_take
from core.broker_alpaca import AlpacaBroker

st.set_page_config(page_title="Invest AI — Secure", layout="wide")

# ---------- Basic Auth Gate ----------
def check_auth():
    import streamlit as st
    pwd_secret = st.secrets.get("APP_PASSWORD", "")
    if not pwd_secret:
        return True  # no password set
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if st.session_state.auth_ok:
        return True
    st.title("🔐 Giriş")
    pwd = st.text_input("Şifrə", type="password")
    if st.button("Daxil ol"):
        if pwd == pwd_secret:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Şifrə yanlışdır.")
    st.stop()

check_auth()

st.title("📈 Invest AI — No‑Code Ticarət Analitikası")

with st.sidebar:
    st.header("⚙️ Parametrlər")
    symbols = st.text_input("Simvollar (vergüllə)", value="AAPL,MSFT,SPY")
    start = st.date_input("Başlanğıc", value=date(2018,1,1))
    end = st.date_input("Son", value=date.today())
    st.subheader("Strategiya")

# 🔮 AI Model seçimi (vizual təkmilləşdirilmiş)
model_options = {
    "⚡ GPT-4o-mini": "Sürətli və ucuz — qısa analizlər üçün ideal",
    "🧠 GPT-4o": "Balanslı və etibarlı — orta səviyyəli strategiyalar üçün",
    "💎 GPT-5": "Ən güclü və analitik — dərin bazar proqnozları üçün"
}

selected_label = st.selectbox(
    "AI modelini seçin:",
    list(model_options.keys()),
    index=0
)
model_map = {
    "⚡ GPT-4o-mini": "gpt-4o-mini",
    "🧠 GPT-4o": "gpt-4o",
    "💎 GPT-5": "gpt-5",
}
openai_model = model_map[selected_label]

st.markdown(f"""
**Aktiv model:** {selected_label}  
_{model_options[selected_label]}_
""")

st.caption(f"**Aktiv model:** {openai_model}")

    rsi_low = st.number_input("RSI aşağı", value=30, step=1)
    rsi_high = st.number_input("RSI yuxarı", value=70, step=1)
    fast_ma = st.number_input("Sürətli MA", value=10, step=1)
    slow_ma = st.number_input("Yavaş MA", value=50, step=1)
    horizon = st.number_input("ML üfüqü (gün)", value=5, step=1)
    test_size = st.slider("Test payı", 0.05, 0.5, 0.2, 0.05)
    st.markdown("---")
    st.subheader("Risk")
    init_cash = st.number_input("Başlanğıc kapital", value=100000, step=1000)
    per_trade_risk = st.number_input("Hər əməliyyat riski", value=0.01, step=0.005, format="%.3f")
    max_pos_pct = st.number_input("Max alət payı", value=0.20, step=0.05, format="%.2f")
    sl_mult = st.number_input("SL (ATR x)", value=2.0, step=0.5, format="%.2f")
    tp_mult = st.number_input("TP (ATR x)", value=4.0, step=0.5, format="%.2f")
    st.markdown("---")
    paper_trade = st.checkbox("Alpaca ilə paper trade", value=False)
    run_btn = st.button("🚀 Analizi işə sal")

if run_btn:
    log_action('run', {'symbols': symbols, 'start': str(start), 'end': str(end)})
    tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    portfolio_equity = []
    per_symbol_rows = []

    for sym in tickers:
        with st.spinner(f"{sym} yüklənir..."):
            df = load_ohlcv(sym, str(start), str(end), interval=interval)
            if df.empty:
                st.warning(f"{sym} üçün məlumat yoxdur.")
                continue

            df_feat = make_tech_features(df)
            model, meta = train_model(df_feat, horizon_days=int(horizon), test_size=float(test_size))
            action, score = generate_signal(df_feat, model, meta,
                                            rsi_bounds=(int(rsi_low), int(rsi_high)),
                                            fast_ma=int(fast_ma), slow_ma=int(slow_ma), ml_weight=0.5)
            price = float(df_feat["Close"].iloc[-1])
            atr = float(df_feat["atr14"].iloc[-1])
            qty = 0
            if action == "BUY":
                qty = position_size(init_cash, price, atr, per_trade_risk, sl_mult, max_pos_pct)
            sl, tp = stop_take(price, atr, sl_mult, tp_mult)

            per_symbol_rows.append({
                "Symbol": sym, "Action": action, "Score": round(float(score),3),
                "Price": round(price, 4), "ATR": round(atr, 4), "Qty": int(qty),
                "SL": round(sl, 4), "TP": round(tp, 4), "ModelAcc": round(meta["acc"],3)
            })

            # Chart
            tab1, tab2 = st.tabs([f"{sym} Chart", f"{sym} Features"])
            with tab1:
                st.line_chart(df["Close"][-300:])
            with tab2:
                st.dataframe(df_feat.tail(10))

            # Paper trade
            if paper_trade and action=="BUY" and qty>0:
                try:
                    broker = AlpacaBroker(paper=True)
                    resp = broker.buy(sym, int(qty)); log_action('order', {'symbol': sym, 'qty': int(qty), 'resp': str(resp)})
                    st.success(f"Alpaca order göndərildi: {resp.get('id','OK')}")
                except Exception as e:
                    st.error(f"Alpaca xətası: {e}")

    if per_symbol_rows:
        st.subheader("🔎 Siqnallar")
        st.dataframe(pd.DataFrame(per_symbol_rows))
        st.info("Qeyd: Bu nəticələr təhsil məqsədlidir. Riskləri özünüz qiymətləndirin.")
else:
    st.write("Sol paneldə parametrləri seç və **Analizi işə sal** düyməsinə bas.")

# ---------- In-app Assistant (Chat) ----------
st.markdown("---")
st.header("🤝 Daxili köməkçi (Chat)")

if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role":"system","content":"Sən Invest AI sisteminin daxili köməkçisisən. İstifadəçiyə strategiya, risk, parametr tənzimləməsi, backtest nəticələrinin izahı, Alpaca inteqrasiyası, Streamlit istifadəsi və ümumi texniki suallarda kömək et. Qısa, konkret cavabla."},
        {"role":"assistant","content":"Salam! Invest AI panelinə xoş gəldin. Parametrləri necə tənzimləmək istəyirsən?"}
    ]

for m in st.session_state.chat:
    with st.chat_message("assistant" if m["role"]=="assistant" else "user"):
        st.markdown(m["content"])

user_msg = st.chat_input("Sualını yaz... (məs: RSI limitlərini necə seçək?)")
if user_msg:
    st.session_state.chat.append({"role":"user","content":user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # call OpenAI
    try:
        import openai, os
        openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
        model = openai_model
        if not openai.api_key:
            raise RuntimeError("OPENAI_API_KEY yoxdur. Streamlit Secrets-dən əlavə edin.")
        from openai import OpenAI
        client = OpenAI(api_key=openai.api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=st.session_state.chat,
            temperature=0.2,
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        reply = f"Chat xətası: {e}"

    st.session_state.chat.append({"role":"assistant","content":reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
