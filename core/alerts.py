# core/alerts.py
import os, requests

def _env(k): 
    return os.getenv(k) or os.environ.get(k)

BOT  = _env("TELEGRAM_BOT_TOKEN")
CHAT = _env("TELEGRAM_CHAT_ID")

def send_telegram(text: str) -> bool:
    if not BOT or not CHAT:
        return False
    url = f"https://api.telegram.org/bot{BOT}/sendMessage"
    r = requests.post(url, json={"chat_id": CHAT, "text": text, "parse_mode": "HTML"}, timeout=10)
    return r.ok
