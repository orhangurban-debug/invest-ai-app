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

# ---------- Alerts: helper-lər ----------
def _init_alert_state():
    if "last_alert_at" not in st.session_state:
        st.session_state.last_alert_at = {}  # { "AAPL": timestamp, ... }

def _rate_limit_ok(sym: str, cooldown_min: int) -> bool:
    """Eyni simvol üçün cooldown pəncərəsi saxla."""
    _init_alert_state()
    import time
    now = time.time()
    last = st.session_state.last_alert_at.get(sym, 0)
    if now - last >= cooldown_min * 60:
        st.session_state.last_alert_at[sym] = now
        return True
    return False

def _log_alert(payload: dict):
    """Göndərilən xəbərdarlıqları fayla yaz."""
    import csv, os, datetime, json as _json
    os.makedirs("logs", exist_ok=True)
    with open("logs/alerts.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([datetime.datetime.utcnow().isoformat(), _json.dumps(payload, ensure_ascii=False)])

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

    # ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔔 Real-time & Alerts")

    auto_refresh = st.checkbox("Auto refresh", value=False, help="Paneli periodik yenilə")
    refresh_sec  = st.number_input("Refresh interval (s)", min_value=5, max_value=600, value=30, step=5)

    enable_tg_alerts = st.checkbox("Telegram Alerts aktiv", value=False)
    alert_prob_th    = st.slider("Prob↑ həd (ML)", 50, 90, 65, 1)          # ML ehtimal %
    alert_er_th      = st.slider("ExpRet həd (%, ML)", 0.0, 10.0, 1.0, 0.1) # ML gözlənilən gəlir %
    alert_score_th   = st.slider("Tech Score həd (0..100)", 0, 100, 60, 1)  # Texniki Score (0..100)

    ai_explain_alert = st.checkbox("AI şərhi ilə birlikdə göndər", value=True)
    alert_cooldown_m = st.number_input("Cooldown (dəq)", 1, 120, 15, 1,
                                       help="Eyni simvol üçün nə qədər tez-tez xəbərdarlıq göndərilsin")

# ================== AUTO-REFRESH (sidebar-dan sonra!) ==================
# Burada artıq auto_refresh və refresh_sec dəyərləri mövcuddur
try:
    if auto_refresh:
        # Streamlit 1.28+ üçün
        try:
            st.autorefresh(interval=int(refresh_sec) * 1000, key="auto_refresh_key")
        except Exception:
            # Köhnə versiyada bu funksiya yoxdursa, sadəcə məlumat veririk
            st.info("Auto-refresh funksiyası bu Streamlit versiyasında mövcud deyil. Manual yenilə.")
except Exception:
    pass

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
                        "Horizon(d)": int(horizon_days),
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

            # --- REALTIME ALERT TRIGGER (ML + Tech) ---
            try:
                if rows_fx and enable_tg_alerts:
                    alerts = []
                    for r in rows_fx:
                        sym   = r["Symbol"]
                        prob  = float(r["Prob↑(%)"])
                        eret  = float(r["ExpRet(%)"])
                        tech_row = df_signals[df_signals["Symbol"] == sym]
                        tech_score = float(tech_row["Score"].iloc[0]) if not tech_row.empty else 0.0

                        if (prob >= alert_prob_th) or (eret >= alert_er_th) or (tech_score >= alert_score_th):
                            if _rate_limit_ok(sym, int(alert_cooldown_m)):
                                alerts.append({
                                    "symbol": sym,
                                    "prob": prob,
                                    "exp_ret": eret,
                                    "tech_score": tech_score,
                                    "reco": r["Recommendation"],
                                    "horizon": r["Horizon(d)"]
                                })

                    if alerts:
                        lines = ["<b>🔔 AI Alert</b>"]
                        for a in alerts:
                            lines.append(
                                f"{a['symbol']}: Prob↑ {a['prob']}% | ExpRet {a['exp_ret']}% | "
                                f"Score {a['tech_score']} | Rec {a['reco']} | H{a['horizon']}d"
                            )

                        if ai_explain_alert:
                            try:
                                ai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
                                if ai_key:
                                    client = OpenAI(api_key=ai_key)
                                    prompt = "Aşağıdakı siqnallar üçün 2-3 cümləlik risk-yönümlü qısa şərh yaz:\n" + "\n".join(lines[1:])
                                    resp = client.chat.completions.create(
                                        model=openai_model,
                                        temperature=0.2,
                                        messages=[
                                            {"role":"system","content":"Qısa, konkret, risk xəbərdarlığı olan peşəkar treyder şərhi ver. Məsləhət deyil."},
                                            {"role":"user","content": prompt}
                                        ]
                                    )
                                    lines.append("\n<b>AI note:</b> " + resp.choices[0].message.content)
                            except Exception as ee:
                                st.warning(f"AI explain alınmadı: {ee}")

                        msg = "\n".join(lines)
                        ok = send_telegram(msg)
                        _log_alert({"alerts": alerts, "sent": ok})
                        st.success("Telegram alert göndərildi ✅" if ok else "Telegram göndərmədi ❗️")
            except Exception as e:
                st.error(f"Alert trigger xətası: {e}")

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

        # --- AUTO TRADE (Alpaca və ya Simulyasiya) ---
        with st.expander("🤖 Auto Trade Executor", expanded=False):
            trade_mode = st.radio("Rejim:", ["Simulyasiya (demo)", "Alpaca Paper Trading"], index=0)
            execute_btn = st.button("🚀 Əməliyyatları yerinə yetir")

            if execute_btn:
                try:
                    results = []
                    for _, row in df_signals.iterrows():
                        sym = row["Symbol"]
                        act = row["Action"]
                        qty = int(row["Qty"])
                        entry = float(row["Entry"])
                        if qty <= 0:
                            continue

                        if trade_mode.startswith("Simulyasiya"):
                            res = simulate_trade(act, sym, qty, entry)
                        else:
                            res = alpaca_trade(act, sym, qty, entry)
                        results.append(res)

                    if results:
                        st.success(f"{len(results)} əməliyyat uğurla icra olundu ✅")
                        st.json(results)
                    else:
                        st.info("Aktiv əməliyyat yoxdur və ya Qty=0.")
                except Exception as e:
                    st.error(f"Auto-trade xətası: {e}")

        # --- TELEGRAM ALERT (manual) ---
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
        
# ================== ALERT MONITOR ==================
st.markdown("---")
st.header("📊 Alert Monitor & History")

import csv

alert_file = "logs/alerts.csv"

if os.path.exists(alert_file):
    df_alerts = pd.read_csv(alert_file, header=None, names=["Time", "Data"])
    df_alerts["Time"] = pd.to_datetime(df_alerts["Time"])
    df_alerts = df_alerts.sort_values("Time", ascending=False)
    
    # Filtrləmə
    c1, c2 = st.columns([2,1])
    recent_only = c2.checkbox("Yalnız son 10 xəbərdarlıq", value=True)
    if recent_only:
        df_alerts = df_alerts.head(10)

    st.dataframe(df_alerts[["Time", "Data"]], use_container_width=True)

    # Ətraflı baxış
    with st.expander("📄 Ətraflı JSON görünüşü", expanded=False):
        for _, row in df_alerts.iterrows():
            st.markdown(f"**🕒 {row['Time']}**")
            try:
                parsed = json.loads(row["Data"])
                st.json(parsed)
            except Exception:
                st.write(row["Data"])
            st.markdown("---")
else:
    st.info("Hələ ki, heç bir xəbərdarlıq (alert) qeydə alınmayıb.")

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
