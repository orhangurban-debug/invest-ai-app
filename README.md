# Invest AI — Streamlit (No‑Code Web App)

Bu paketlə **kod bilmədən** brauzerdən işləyən ticarət analitika paneli quracaqsınız.
- Portfel backtest, walk-forward qiymətləndirmə
- Siqnal: qayda (RSI/MA) + ML (RandomForest, default)
- Risk parametrləri (ATR stop, per-trade risk, günlük loss stop)
- **Paper trade (Alpaca)** — isteğe bağlı
- Heç bir lokal quraşdırma vacib deyil: **Streamlit Cloud** üzərinə deploy edin

## Tez Başlanğıc (Streamlit Cloud)
1. Bu layihəni ZIP-dən çıxarın və **GitHub**-da yeni repo yaradıb faylları oraya yükləyin.
2. https://share.streamlit.io → "Deploy an app" → GitHub repo-nuzu seçin → `app.py` faylını göstərin.
3. **Secrets** əlavə edin (Əgər Alpaca istifadə edəcəksinizsə): Settings → Secrets →
```
ALPACA_API_KEY_ID="..."
ALPACA_API_SECRET_KEY="..."
```
4. Deploy edildikdən sonra URL brauzerdə açılacaq.

## Lokal İşlətmək (opsional)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
streamlit run app.py
```

## Fayllar
- `app.py` — Streamlit UI
- `core/*.py` — analitika modulları
- `requirements.txt` — asılılıqlar
- `.streamlit/secrets.toml.example` — nümunə secrets


---

## 🔐 Təhlükəsizlik və İnteqrasiya qeydləri (ABB/Brókerlər)
- **Giriş qoruması:** `APP_PASSWORD` ilə sadə giriş pəncərəsi əlavə olunub. Streamlit Secrets-də saxlayın.
- **Sirlər (Secrets):** OPENAI və Alpaca açarlarını yalnız Secrets-də saxlayın.
- **Jurnallar:** `logs/actions.csv` istifadəçi hərəkətlərini qeyd edir (lokal deploy üçün).
- **Model idarəsi:** UI-dan model adı seçimi (secrets-dən default).
- **CI/CD və Auto‑Update:** Kodu GitHub-a push etdikcə Streamlit Cloud avtomatik yenilənir.
- **ABB inteqrasiyası:** ABB‑nin “Business API” bank əməliyyatları üçündür, ticarət əmrləri üçün birbaşa API rəsmi şəkildə dərc edilməyib. ABB‑Invest hazırda MT4 (forex/CFD) və ABB mobil app vasitəsilə ticarət təklif edir. Bu panel icra üçün **Alpaca/IBKR** kimi broker API-lərlə işləmək üçündür; ABB ilə birbaşa ticarət üçün ABB‑Invest ilə API əməkdaşlığı tələb oluna bilər.

