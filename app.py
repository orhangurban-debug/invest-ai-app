# app.py
import os, json, datetime
from datetime import date

import pandas as pd
import streamlit as st

# ---- Daxili modullar ----
from core.features import add_indicators
from core.strategy import latest_signal
from core.risk import position_size, make_trade_plan
from core.trade_log import append_trade, read_log
from core.alerts import send_telegram
from core.charts import price_chart
from core.backtest import run_backtest
from openai import OpenAI
from core.predictor import ai_forecast  # ML proqnoz üçün
from core.portfolio import analyze_portfolio, optimize_allocation, build_trade_plan, calc_var_es, calc_sharpe
from core.trader import alpaca_trade, simulate_trade

# ---- Səhifə konfiqurasiyası ----
st.set_page_config(page_title="Invest AI — Secure", layout="wide")

# ================== CACHE-Lİ YÜKLƏMƏ ==================
@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_many(symbol_list, start, end, interval):
    from core.data import load_many as _load_many
    return _load_many(symbol_list, start, end, interval)

# ================== LOG HELPER ==================
def log_action(kind: str, payload: dict):
    import csv
    os.makedirs("logs", exist_ok=True)
    with open("logs/actions.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            datetime.datetime.utcnow().isoformat(),
            kind,
            json.dumps(payload, ensure_ascii=False)
        ])

# ================== BASIC AUTH ==================
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

# ================== UI: BAŞLIQ ==================
check_auth()
st.title("📈 Invest AI — No-Code Ticarət Analitikası")

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("⚙ Parametrlər")

    symbols  = st.text_input("Simvollar (vergüllə)", value="AAPL,MSFT,SPY")
    start    = st.date_input("Başlanğıc", value=date(2020, 1, 1))
    end      = st.date_input("Son", value=date.today())
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    st.subheader("Strategiya")
    model_options = {
        "⚡ GPT-4o-mini": "Sürətli və ucuz — qısa analizlər üçün ideal",
        "🧠 GPT-4o":      "Balanslı və etibarlı — orta səviyyəli strategiyalar üçün",
        "💎 GPT-5":       "Ən güclü və analitik — dərin bazar proqnozları üçün"
    }
    selected_label = st.selectbox("AI modelini seçin:", list(model_options.keys()), index=0)
    model_map = {
        "⚡ GPT-4o-mini": "gpt-4o-mini",
        "🧠 GPT-4o": "gpt-4o",
        "💎 GPT-5": "gpt-5"
    }
    openai_model = model_map[selected_label]
    st.markdown(f"**Aktiv model:** {selected_label}\n\n_{model_options[selected_label]}_")
    st.caption(f"**Aktiv model kodu:** {openai_model}")

    st.subheader("Bildiriş və hədəflər")
    alert_score_up = st.slider("Alert skoru (↑)", 50, 90, 60, 1)
    atr_mult_sl    = st.number_input("SL (ATR x)", value=2.0, step=0.5, format="%.1f")
    atr_mult_tp    = st.number_input("TP (ATR x)", value=3.0, step=0.5, format="%.1f")

    st.subheader("Texniki parametrlər")
    rsi_low  = st.number_input("RSI aşağı",  value=30, step=1)
    rsi_high = st.number_input("RSI yuxarı", value=70, step=1)
    fast_ma  = st.number_input("Sürətli MA", value=10, step=1)
    slow_ma  = st.number_input("Yavaş MA",   value=50, step=1)

    st.subheader("Risk")
    init_cash      = st.number_input("Başlanğıc kapital", value=100000, step=1000)
    per_trade_risk = st.number_input("Hər əməliyyat riski", value=0.01, step=0.005, format="%.3f")

    st.markdown("---")
    st.subheader("ML Forecast parametrləri")
    horizon_days   = st.slider("Proqnoz üfüqü (gün)", 3, 20, 5, 1)
    ml_model_type  = st.selectbox("ML model növü", ["xgb", "rf"], index=0)
    max_pos_pct    = st.number_input("Max alət payı", value=0.25, step=0.05, format="%.2f")

# ================== MAIN: LIVE SIGNALS ==================
st.markdown("## 🔎 Live Signals")
run_btn = st.button("🚀 Analizi işə sal")

if run_btn:
    log_action("run", {"symbols": symbols, "start": str(start), "end": str(end), "interval": interval})
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    try:
        raw = cached_load_many(symbol_list, str(start), str(end), interval)
    except Exception as e:
        st.error(f"Data xətası: {e}")
        raw = {}

    rows = []
    for sym, df in raw.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            st.warning(f"{sym}: məlumat tapılmadı və ya boş DataFrame.")
            continue

        f = add_indicators(df)
        if f.empty:
            st.warning(f"{sym}: göstəricilər yaradıla bilmədi (tarix çox qısadır və ya NaN çoxdur).")
            continue

        score, action, last = latest_signal(f)

        atr_val = last.get("atr") or last.get("atr14") or last.get("ATR")
        if atr_val is None:
            atr_val = float(last["close"]) * 0.02

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

        # --- AI FORECAST ---
        with st.expander("🤖 AI Forecast (Expected Return & Recommendation)", expanded=False):
            try:
                rows_fx = []
                for sym in df_signals["Symbol"]:
                    df_raw = raw.get(sym)
                    if not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
                        continue
                    fx = ai_forecast(df_raw, horizon_days=int(horizon_days), model_type=ml_model_type)
                    rows_fx.append({
                        "Symbol": sym,
                        "Prob↑(%)": round(fx["prob_up"] * 100, 1),
                        "ExpRet(%)": round(fx["expected_return"] * 100, 2),
                        "Model Acc": round(fx["acc"] * 100, 1),
                        "Recommendation": fx["recommendation"]
                    })
                if rows_fx:
                    st.dataframe(pd.DataFrame(rows_fx).sort_values("ExpRet(%)", ascending=False),
                                 use_container_width=True)
                else:
                    st.info("Forecast üçün məlumat azdır.")
            except Exception as e:
                st.error(f"Forecast xətası: {e}")

        # --- QRAFİK (TOP 2) ---
        with st.expander("📈 Qrafik (Top 2 siqnal)", expanded=False):
            for sym in df_signals["Symbol"].head(2):
                df_raw = raw.get(sym)
                if isinstance(df_raw, pd.DataFrame) and not df_raw.empty:
                    st.plotly_chart(price_chart(df_raw, title=sym), use_container_width=True)

        # --- PORTFOLIO & RISK DASHBOARD ---
        with st.expander("💼 Portfolio Analysis & Risk Metrics", expanded=False):
            try:
                if 'df_signals' not in locals() or df_signals.empty:
                    st.info("Portfel analizi üçün siqnal cədvəli yoxdur.")
                else:
                    syms_for_pf = list(df_signals["Symbol"].values)
                    analysis = analyze_portfolio(raw, syms_for_pf, horizon_days=int(horizon_days))
                    rows_pf = analysis["rows"]

                    if not rows_pf:
                        st.info("Portfel analizi üçün kifayət qədər data yoxdur.")
                    else:
                        df_pf = pd.DataFrame(rows_pf).sort_values("exp_ret", ascending=False)
                        st.dataframe(df_pf.rename(columns={
                            "symbol":"Symbol","exp_ret":"ExpRet(d)","vol":"Vol(d)","prob_up":"Prob↑","acc":"ModelAcc"
                        }), use_container_width=True)

                        candidates = [{"symbol": r["symbol"], "exp_ret": r["exp_ret"], "vol": r["vol"]} for r in rows_pf]
                        weights = optimize_allocation(candidates, max_pos_pct=float(max_pos_pct))

                        entries = {r["Symbol"]: float(r["Entry"]) for _, r in df_signals.iterrows()}
                        atrs = {}
                        for sym in syms_for_pf:
                            df_raw = raw.get(sym)
                            if isinstance(df_raw, pd.DataFrame) and not df_raw.empty:
                                atrs[sym] = float((df_raw["high"] - df_raw["low"]).rolling(14).mean().dropna().iloc[-1])

                        plans = build_trade_plan(weights=weights, entries=entries, atrs=atrs,
                                                 init_cash=float(init_cash), per_trade_risk=float(per_trade_risk))

                        plan_rows = []
                        for sym, p in plans.items():
                            plan_rows.append({
                                "Symbol": sym, "Weight": p["weight"], "Target $": p["target_cash"],
                                "Entry": p["entry"], "SL": p["sl"], "ATR": p["atr"], "Qty": p["qty"]
                            })
                        st.subheader("🔧 AI Allocation — Hədəf bölüşdürmə")
                        st.dataframe(pd.DataFrame(plan_rows).sort_values("Weight", ascending=False), use_container_width=True)

                        port_ret = None
                        for sym, w in weights.items():
                            r = analysis["returns"].get(sym)
                            if r is None or r.empty:
                                continue
                            r_w = r * float(w)
                            port_ret = r_w if port_ret is None else port_ret.add(r_w, fill_value=0)

                        if port_ret is not None and len(port_ret) > 50:
                            var95, es95 = calc_var_es(port_ret.dropna(), alpha=0.95)
                            sharpe = calc_sharpe(port_ret.dropna())
                            c1, c2, c3 = st.columns(3)
                            c1.metric("VaR (95%, gündəlik)", f"{var95*100:.2f}%")
                            c2.metric("ES (95%, gündəlik)", f"{es95*100:.2f}%")
                            c3.metric("Sharpe (illik)", f"{sharpe:.2f}")
                        else:
                            st.info("VaR/ES/Sharpe üçün kifayət qədər tarixi məlumat yoxdur.")
            except Exception as e:
                st.error(f"Portfolio xətası: {e}")

        # --- BACKTEST ---
        st.markdown("---")
        with st.container():
            st.subheader("🧪 Backtest (sadə qayda ilə)")
            if st.button("Backtest-i işə sal"):
                for sym in df_signals["Symbol"]:
                    df_raw = raw.get(sym)
                    if isinstance(df_raw, pd.DataFrame) and not df_raw.empty:
                        bt = run_backtest(df_raw, rsi_low=rsi_low, rsi_high=rsi_high,
                                          fast_ma=fast_ma, slow_ma=slow_ma)
                        col1, col2 = st.columns(2)
                        col1.metric(f"{sym} — Total Return", f"{bt['total_return']*100:.1f}%")
                        col2.metric(f"{sym} — Max DD", f"{bt['max_drawdown']*100:.1f}%")
                        st.line_chart(bt["equity_curve"])

        # --- TELEGRAM ALERT ---
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

else:
    st.info("Sol paneldə parametrləri seç və **Analizi işə sal** düyməsinə bas.")

# ================== IN-APP CHAT ==================
st.markdown("---")
st.header("🤝 Daxili köməkçi (Chat)")

if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "system",
         "content": "Sən Invest AI sisteminin daxili köməkçisisən. İstifadəçiyə strategiya, risk, parametr tənzimləməsi, backtest nəticələrinin izahı, Alpaca inteqrasiyası, Streamlit istifadəsi və ümumi texniki suallarda kömək et. Qısa, konkret cavabla."},
        {"role": "assistant",
         "content": "Salam! Invest AI panelinə xoş gəldin. Parametrləri necə tənzimləmək istəyirsən?"}
    ]

for m in st.session_state.chat:
    with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
        st.markdown(m["content"])

user_msg = st.chat_input("Sualını yaz... (məs: RSI limitlərini necə seçək?)")
if user_msg:
    st.session_state.chat.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)
    try:
        ai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
        if not ai_key:
            raise RuntimeError("OPENAI_API_KEY yoxdur. Settings → Secrets bölməsinə əlavə edin.")
        client = OpenAI(api_key=ai_key)
        resp = client.chat.completions.create(
            model=openai_model,
            messages=st.session_state.chat,
            temperature=0.2,
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        reply = f"Chat xətası: {e}"
    st.session_state.chat.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

# ================== TRADE LOG ==================
st.markdown("## 📒 Trade Log")
log_df = read_log()
st.dataframe(log_df, use_container_width=True)

with st.expander("➕ Əməliyyatı jurnala əlavə et"):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    sym   = c1.text_input("Symbol")
    act   = c2.selectbox("Action", ["Buy", "Sell", "Exit", "Adjust SL"])
    entry = c3.number_input("Entry", value=0.0, step=0.01)
    sl    = c4.number_input("SL", value=0.0, step=0.01)
    tp    = c5.number_input("TP", value=0.0, step=0.01)
    qty   = c6.number_input("Qty", value=0, step=1)
    note  = st.text_input("Qeyd")

    if st.button("Jurnala yaz"):
        append_trade({
            "symbol": sym,
            "action": act,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "qty": qty,
            "score": None,
            "note": note
        })
        st.rerun()
