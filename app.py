import os, time, json, datetime
import pandas as pd
import streamlit as st
from datetime import date

# --- daxili modul importları (YEKCINS VERZIYA) ---
from core.data import load_many
from core.features import add_indicators
from core.strategy import latest_signal
from core.risk import position_size, make_trade_plan
from core.trade_log import append_trade, read_log
from core.alerts import send_telegram

st.set_page_config(page_title="Invest AI — Secure", layout="wide")

# ---------- Log helper ----------
def log_action(kind, payload: dict):
    import csv, os, datetime
    os.makedirs("logs", exist_ok=True)
    with open("logs/actions.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([datetime.datetime.utcnow().isoformat(), kind, json.dumps(payload, ensure_ascii=False)])

# ---------- Basic Auth Gate ----------
def check_auth():
    pwd_secret = st.secrets.get("APP_PASSWORD", "")
    if not pwd_secret:
        return True
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

st.title("📈 Invest AI — No-Code Ticarət Analitikası")

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("⚙ Parametrlər")

    # Tarix və simvollar
    symbols = st.text_input("Simvollar (vergüllə)", value="AAPL,MSFT,SPY")
    start   = st.date_input("Başlanğıc", value=date(2018, 1, 1))
    end     = st.date_input("Son", value=date.today())
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    st.subheader("Strategiya")

    # AI model seçimi (sənin vizual seçimlərinlə)
    model_options = {
        "⚡ GPT-4o-mini": "Sürətli və ucuz — qısa analizlər üçün ideal",
        "🧠 GPT-4o": "Balanslı və etibarlı — orta səviyyəli strategiyalar üçün",
        "💎 GPT-5": "Ən güclü və analitik — dərin bazar proqnozları üçün"
    }
    selected_label = st.selectbox("AI modelini seçin:", list(model_options.keys()), index=0)
    model_map = {"⚡ GPT-4o-mini": "gpt-4o-mini", "🧠 GPT-4o": "gpt-4o", "💎 GPT-5": "gpt-5"}
    openai_model = model_map[selected_label]
    st.markdown(f"**Aktiv model:** {selected_label}  \n_{model_options[selected_label]}_")
    st.caption(f"**Aktiv model kodu:** {openai_model}")

    # Bildiriş və hədəflər
    alert_score_up = st.slider("Alert skoru (↑)", 50, 90, 60, 1)
    atr_mult_sl    = st.number_input("SL (ATR x)", value=2.0, step=0.5, format="%.1f")
    atr_mult_tp    = st.number_input("TP (ATR x)", value=3.0, step=0.5, format="%.1f")

    # Texniki parametrlər
    rsi_low  = st.number_input("RSI aşağı",  value=30, step=1)
    rsi_high = st.number_input("RSI yuxarı", value=70, step=1)
    fast_ma  = st.number_input("Sürətli MA", value=10, step=1)
    slow_ma  = st.number_input("Yavaş MA",   value=50, step=1)

    st.markdown("---")
    st.subheader("Risk")
    init_cash     = st.number_input("Başlanğıc kapital", value=100000, step=1000)
    per_trade_risk = st.number_input("Hər əməliyyat riski", value=0.01, step=0.005, format="%.3f")

# ===================== MAIN: LIVE SIGNALS =====================
st.markdown("## 🔎 Live Signals")
run_btn = st.button("🚀 Analizi işə sal")

if run_btn:
    log_action('run', {'symbols': symbols, 'start': str(start), 'end': str(end), 'interval': interval})
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    try:
        raw = load_many(symbol_list, str(start), str(end), interval)
    except Exception as e:
        st.error(f"Data xətası: {e}")
        raw = {}

    rows = []
    for sym, df in raw.items():
        # ✅ DF yoxlaması
        if not isinstance(df, pd.DataFrame) or df.empty:
            st.warning(f"{sym}: məlumat tapılmadı və ya boş DataFrame.")
            continue

        f = add_indicators(df)
        if f.empty:
            st.warning(f"{sym}: göstəricilər yaradıla bilmədi (tarix çox qısadır və ya NaN çoxdur).")
            continue

        score, action, last = latest_signal(f)

        # ATR təhlükəsiz oxu
        atr_val = last.get("atr") or last.get("atr14") or last.get("ATR")
        if atr_val is None:
            atr_val = float(last["close"]) * 0.02  # ehtiyat dəyər

        entry, sl, tp = make_trade_plan(
            float(last["close"]), float(atr_val),
            atr_mult_sl=float(atr_mult_sl),
            atr_mult_tp=float(atr_mult_tp)
        )
        qty = position_size(float(init_cash), float(per_trade_risk), entry, sl)
        rr  = round((tp - entry) / max(entry - sl, 0.001), 2)

        rows.append({
            "Symbol": sym, "Score": round(float(score), 1), "Action": action,
            "Entry": entry, "SL": sl, "TP": tp, "Qty": qty, "R:R": rr
        })

    if not rows:
        st.warning("Analiz üçün məlumat tapılmadı.")
    else:
        df_signals = pd.DataFrame(rows).sort_values("Score", ascending=False)
        st.dataframe(df_signals, use_container_width=True)

        if st.button("🔔 Telegram (Score ≥ seçilmiş hədd)"):
            msg = ["<b>Live Signals</b>"]
            for r in rows:
                if r["Score"] >= alert_score_up:
                    msg.append(
                        f"{r['Symbol']}: <b>{r['Action']}</b> | {r['Entry']} / SL {r['SL']} / TP {r['TP']} | "
                        f"Qty {r['Qty']} | Score {r['Score']} | R:R {r['R:R']}"
                    )
            ok = send_telegram("\n".join(msg)) if len(msg) > 1 else False
            st.success("Bildiriş göndərildi ✅" if ok else "Siqnal yoxdur və ya Telegram secrets boşdur ❗️")

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
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
        model = openai_model
        if not openai.api_key:
            raise RuntimeError("OPENAI_API_KEY yoxdur. Settings → Secrets bölməsinə əlavə edin.")
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

# ---------- Trade Log ----------
st.markdown("## 📒 Trade Log")
log_df = read_log()
st.dataframe(log_df, use_container_width=True)

with st.expander("➕ Əməliyyatı jurnala əlavə et"):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    sym   = c1.text_input("Symbol")
    act   = c2.selectbox("Action", ["Buy","Sell","Exit","Adjust SL"])
    entry = c3.number_input("Entry", value=0.0, step=0.01)
    sl    = c4.number_input("SL", value=0.0, step=0.01)
    tp    = c5.number_input("TP", value=0.0, step=0.01)
    qty   = c6.number_input("Qty", value=0, step=1)
    note  = st.text_input("Qeyd")

    if st.button("Jurnala yaz"):
        append_trade({
            "symbol": sym, "action": act, "entry": entry,
            "sl": sl, "tp": tp, "qty": qty, "score": None, "note": note
        })
        st.rerun()
