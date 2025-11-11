# core/trader.py
import os, requests, datetime

def alpaca_trade(action: str, symbol: str, qty: int, entry: float):
    """
    Alpaca API ilə əməliyyat icrası.
    """
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    api_key = os.getenv("ALPACA_API_KEY_ID", "")
    api_secret = os.getenv("ALPACA_SECRET_KEY", "")
    if not api_key or not api_secret:
        raise RuntimeError("Alpaca API açarları yoxdur. Secrets → ALPACA_API_KEY_ID, ALPACA_SECRET_KEY əlavə et.")

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret
    }

    side = "buy" if action.lower() == "buy" else "sell"
    data = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": "market",
        "time_in_force": "gtc"
    }

    r = requests.post(f"{base_url}/v2/orders", headers=headers, json=data)
    if r.status_code != 200:
        raise RuntimeError(f"Alpaca xətası: {r.text}")

    return r.json()

def simulate_trade(action: str, symbol: str, qty: int, price: float):
    """
    Offline demo əməliyyat.
    """
    now = datetime.datetime.utcnow().isoformat()
    return {
        "timestamp": now,
        "symbol": symbol,
        "action": action,
        "qty": qty,
        "price": price,
        "status": "executed (simulated)"
    }
