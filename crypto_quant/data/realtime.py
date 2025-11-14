"""Realtime market data utilities."""
from __future__ import annotations

import json
from typing import Optional
from urllib.request import urlopen, Request


def fetch_btc_price_coingecko() -> Optional[float]:
    """Fetch current BTC price in USD from CoinGecko.

    Returns:
        Optional[float]: Price in USD if successful, else None.
    """

    try:
        url = (
            "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        )
        req = Request(url, headers={"User-Agent": "crypto-quant/1.0"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return float(data.get("bitcoin", {}).get("usd"))
    except Exception:
        return None
