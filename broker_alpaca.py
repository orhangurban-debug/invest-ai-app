import os, requests

class AlpacaBroker:
    def __init__(self, paper=True):
        base = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.base = base
        self.key = os.getenv("ALPACA_API_KEY_ID") or ""
        self.secret = os.getenv("ALPACA_API_SECRET_KEY") or ""
        if not self.key or not self.secret:
            raise RuntimeError("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY tapılmadı (Streamlit Secrets).")
        self.hdr = {
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
            "Content-Type": "application/json"
        }
    def _post(self, path, payload):
        r = requests.post(self.base + path, headers=self.hdr, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    def buy(self, symbol: str, qty: int):
        return self._post("/v2/orders", {"symbol": symbol, "qty": qty, "side": "buy",
                                         "type": "market", "time_in_force": "gtc"})
    def sell(self, symbol: str, qty: int):
        return self._post("/v2/orders", {"symbol": symbol, "qty": qty, "side": "sell",
                                         "type": "market", "time_in_force": "gtc"})
