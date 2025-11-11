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
    
# === REAL-TIME MONITOR & PERFORMANCE (paste after Telegram block) =====================
st.markdown("---")
with st.expander("📡 Real-time Monitor & Performance", expanded=False):

    # -------- helpers --------
    def _ensure_raw_for_monitor():
        """Monitor işləsin deyə data olsun: varsa 'raw' istifadə et, yoxdursa minimal yüklə."""
        try:
            return raw  # noqa: F821  # 'raw' varsa (yuxarıda analizi işə salmısınızsa) onu istifadə edir
        except NameError:
            # minimal snapshot (son 2 il), yalnız monitor üçün
            end_dt = date.today()
            start_dt = date(end_dt.year - 2, end_dt.month, end_dt.day)
            syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            try:
                return cached_load_many(syms, str(start_dt), str(end_dt), interval)
            except Exception:
                return {}

    def _last_close(df_: pd.DataFrame) -> float:
        if not isinstance(df_, pd.DataFrame) or df_.empty:
            return float("nan")
        # close sütunu həm 'close', həm də 'Close' ola bilər – hər ikisini yoxlayırıq
        for c in ("close", "Close"):
            if c in df_.columns:
                return float(pd.to_numeric(df_[c], errors="coerce").dropna().iloc[-1])
        return float("nan")

    def _mk_positions_from_log(_log: pd.DataFrame) -> pd.DataFrame:
        """
        Trade Log-a əsasən sadə mövqe xülasəsi:
        - Buy -> +qty, Sell -> -qty
        - Weighted avg cost hesablanır
        """
        if _log is None or _log.empty:
            return pd.DataFrame(columns=["symbol", "qty", "avg_cost"])
        x = _log.copy()
        x["qty"] = pd.to_numeric(x["qty"], errors="coerce").fillna(0).astype(float)
        x["entry"] = pd.to_numeric(x["entry"], errors="coerce").fillna(0).astype(float)
        x["side"] = x["action"].str.lower().map({"buy": 1, "sell": -1}).fillna(0)

        rows = []
        for sym, g in x.groupby("symbol"):
            buys  = g[g["side"] == 1]
            sells = g[g["side"] == -1]
            qty   = float((buys["qty"].sum() - sells["qty"].sum()))
            if qty <= 0:
                continue
            # weighted avg cost yalnız buy-lardan
            if buys.empty:
                avg_cost = 0.0
            else:
                cost = (buys["qty"] * buys["entry"]).sum()
                avg_cost = float(cost / max(buys["qty"].sum(), 1e-9))
            rows.append({"symbol": sym, "qty": qty, "avg_cost": avg_cost})
        return pd.DataFrame(rows)

    # -------- load data for monitor --------
    mon_raw = _ensure_raw_for_monitor()
    log_df  = read_log()

    # canlı qiymətlər
    prices = {s: _last_close(df_) for s, df_ in (mon_raw or {}).items()}

    # mövqelər
    pos_df = _mk_positions_from_log(log_df)

    # PnL hesabı
    pnl_rows = []
    total_unrl = 0.0
    for _, r in pos_df.iterrows():
        sym = r["symbol"]
        qty = float(r["qty"])
        avg = float(r["avg_cost"])
        last = float(prices.get(sym, float("nan")))
        if not pd.isna(last) and qty > 0:
            unrl = (last - avg) * qty
            total_unrl += unrl
            pnl_rows.append({
                "Symbol": sym,
                "Qty": int(qty),
                "Avg cost": round(avg, 4),
                "Last": round(last, 4),
                "Unrealized PnL ($)": round(unrl, 2),
                "Unrealized PnL (%)": round(((last / avg) - 1.0) * 100.0, 2) if avg > 0 else 0.0
            })

    pnl_df = pd.DataFrame(pnl_rows)

    # təxmini sərbəst nağd: sidebar-dakı init_cash minus mövcud mövqelərin dəyəri
    invested_val = 0.0
    for _, r in pos_df.iterrows():
        invested_val += float(r["qty"]) * float(r["avg_cost"])
    try:
        free_cash = float(init_cash) - invested_val  # init_cash sidebar-dandır
    except Exception:
        free_cash = float("nan")

    # -------- UI --------
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Free cash (approx.)", f"${free_cash:,.0f}")
    c2.metric("📈 Unrealized PnL", f"${total_unrl:,.0f}")
    c3.metric("🧾 Open positions", f"{len(pnl_df)}")

    if not pnl_df.empty:
        st.dataframe(pnl_df.sort_values("Unrealized PnL ($)", ascending=False), use_container_width=True)
    else:
        st.info("Açıq mövqe tapılmadı. Mövqelər `Trade Log` vasitəsilə yaranır (Buy/Sell).")

    # mini live-chart: ilk simvol üçün son 200 şam (əgər data var)
    try:
        first_sym = None
        if isinstance(mon_raw, dict) and mon_raw:
            first_sym = next(iter(mon_raw.keys()))
        if first_sym:
            df_raw = mon_raw[first_sym]
            if isinstance(df_raw, pd.DataFrame) and not df_raw.empty:
                st.plotly_chart(price_chart(df_raw.tail(200), title=f"{first_sym} — last 200 candles"),
                                use_container_width=True)
    except Exception:
        pass

    # -------- Auto-refresh (sidebar parametrindən) --------
    try:
        if 'auto_refresh' in locals() and auto_refresh:
            import time
            st.caption(f"♻️ Auto-refresh aktivdir — {int(refresh_sec)} saniyədən bir yenilənəcək.")
            time.sleep(int(refresh_sec))
            st.rerun()
    except Exception:
        # auto-refresh olmasa da səhifə normal işləsin
        pass
# ================================================================================

# === AI-TRIGGERED TELEGRAM ALERTS (paste after Real-time Monitor block) =================
st.markdown("---")
with st.expander("🤖 AI Alert System — Auto Trigger", expanded=False):

    # Parametrlər sidebar-dan gəlir:
    # enable_tg_alerts, alert_prob_th, alert_er_th, alert_score_th,
    # ai_explain_alert, alert_cooldown_m
    if not enable_tg_alerts:
        st.info("Telegram Alerts deaktivdir — sidebar-dan 'Telegram Alerts aktiv' işarələyin.")
    else:
        try:
            # 1) Məlumatların hazır olması (raw/df_signals/ai_forecast üçün)
            if 'df_signals' not in locals() or df_signals.empty:
                st.warning("Alert üçün siqnal cədvəli tapılmadı. Öncə 'Analizi işə sal' düyməsinə basın.")
            else:
                triggers = []     # UI göstərmək üçün
                alert_lines = []  # Telegram mətni
                ai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
                client = OpenAI(api_key=ai_key) if (ai_explain_alert and ai_key) else None

                # 2) Hər simvol üzrə ML + texniki yoxlama
                for _, row in df_signals.iterrows():
                    sym   = row["Symbol"]
                    score = float(row["Score"])
                    dfraw = (raw or {}).get(sym) if 'raw' in locals() else None
                    if not isinstance(dfraw, pd.DataFrame) or dfraw.empty:
                        continue

                    fx = ai_forecast(
                        dfraw,
                        horizon_days=int(horizon_days),
                        model_type=ml_model_type
                    )
                    prob_up = fx["prob_up"] * 100.0
                    exp_ret = fx["expected_return"] * 100.0
                    reco    = fx["recommendation"]
                    acc     = fx["acc"] * 100.0

                    # 3) Trigger şərtləri
                    cond_prob  = (prob_up >= float(alert_prob_th))
                    cond_er    = (exp_ret >= float(alert_er_th))
                    cond_score = (score  >= float(alert_score_th))

                    if cond_prob and cond_er and cond_score:
                        # rate limit (cooldown) – eyni simvol üçün təkrarlamayaq
                        if _rate_limit_ok(sym, int(alert_cooldown_m)):
                            # (i) Telegram mətni
                            line = (f"⚠️ <b>{sym}</b> — <i>Auto Alert</i>\n"
                                    f"Recommendation: <b>{reco}</b>\n"
                                    f"Prob↑: {prob_up:.1f}%  |  ExpRet: {exp_ret:.2f}%  |  Score: {score:.0f}  |  Acc: {acc:.1f}%")
                            # (ii) AI qısa şərh (opsional)
                            if client:
                                try:
                                    brief = (
                                        f"Symbol: {sym}\nProbUp: {prob_up:.1f}%  ExpRet: {exp_ret:.2f}%  "
                                        f"TechScore: {score:.0f}  ModelAcc: {acc:.1f}%  Horizon(d): {int(horizon_days)}\n"
                                        f"TL;DR: 2 cümləlik icmal və 1 cümləlik risk xəbərdarlığı ver."
                                    )
                                    resp = client.chat.completions.create(
                                        model=openai_model,
                                        temperature=0.2,
                                        messages=[
                                            {"role":"system","content":"Təcrübəli risk yönümlü treyder kimi çox qısa və konkret izah ver. MALİYYƏ MƏSLƏHƏTİ DEYİL."},
                                            {"role":"user","content": brief}
                                        ]
                                    )
                                    ai_text = resp.choices[0].message.content.strip()
                                    line += f"\n\n{ai_text}"
                                except Exception as e:
                                    st.warning(f"AI şərh alınmadı: {e}")

                            alert_lines.append(line)
                            triggers.append({
                                "Symbol": sym,
                                "Prob↑(%)": round(prob_up,1),
                                "ExpRet(%)": round(exp_ret,2),
                                "Score": round(score,1),
                                "Reco": reco,
                                "ModelAcc(%)": round(acc,1),
                                "Status": "TRIGGERED ✅"
                            })
                            # Log-a yaz
                            _log_alert({"symbol": sym, "prob_up": prob_up, "exp_ret": exp_ret,
                                        "score": score, "reco": reco, "ts": datetime.datetime.utcnow().isoformat()})
                        else:
                            triggers.append({
                                "Symbol": sym, "Status": f"cooldown {int(alert_cooldown_m)} dəq ⏳"
                            })
                    else:
                        triggers.append({
                            "Symbol": sym,
                            "Prob↑(%)": round(prob_up,1),
                            "ExpRet(%)": round(exp_ret,2),
                            "Score": round(score,1),
                            "Status": "no trigger"
                        })

                # 4) UI: hansı simvolların keçdiyini göstər
                if triggers:
                    st.dataframe(pd.DataFrame(triggers), use_container_width=True)

                # 5) Telegram-a göndər
                if alert_lines:
                    text = "<b>AI Auto Alerts</b>\n" + "\n\n".join(alert_lines)
                    ok = send_telegram(text)
                    st.success("Telegram göndərildi ✅" if ok else "Telegram göndərilə bilmədi ❗️")
                    if not ok:
                        st.caption("Bot token / chat_id secrets bölməsində düzgün deyil.")
                else:
                    st.info("Bu dəfə şərtləri keçən siqnal yoxdur.")
        except Exception as e:
            st.error(f"AI Alert xətası: {e}")
# =======================================================================================

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
