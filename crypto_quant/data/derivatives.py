"""Derivatives market data from Binance public endpoints."""
from __future__ import annotations

import json
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


_BINANCE_FAPI = "https://fapi.binance.com"
_BINANCE_API = "https://api.binance.com"


def _get_json(url: str) -> Optional[dict | list]:
    try:
        req = Request(url, headers={"User-Agent": "crypto-quant/1.0"})
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def fetch_funding_rate(symbol: str = "BTCUSDT", limit: int = 1000) -> pd.DataFrame:
    """Fetch historical funding rates (futures) from Binance.

    Args:
        symbol (str): Binance futures symbol.
        limit (int): Number of entries.

    Returns:
        pd.DataFrame: ['date','fundingRate']
    """

    q = urlencode({"symbol": symbol, "limit": limit})
    url = f"{_BINANCE_FAPI}/fapi/v1/fundingRate?{q}"
    data = _get_json(url)
    if not isinstance(data, list):
        return pd.DataFrame(columns=["date", "fundingRate"])
    df = pd.DataFrame(data)
    if "fundingRate" in df.columns:
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["date"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    return df.dropna(subset=["date", "fundingRate"])[["date", "fundingRate"]].sort_values("date").reset_index(drop=True)


def fetch_open_interest(symbol: str = "BTCUSDT", interval: str = "1d", limit: int = 200) -> pd.DataFrame:
    """Fetch open interest historical data from Binance futures.

    Args:
        symbol (str): Binance symbol.
        interval (str): e.g., '5m','15m','1h','4h','1d'.
        limit (int): Max records.

    Returns:
        pd.DataFrame: ['date','openInterest']
    """

    q = urlencode({"symbol": symbol, "period": interval, "limit": limit})
    url = f"{_BINANCE_FAPI}/futures/data/openInterestHist?{q}"
    data = _get_json(url)
    if not isinstance(data, list):
        return pd.DataFrame(columns=["date", "openInterest"])
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["openInterest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    return df.dropna(subset=["date", "openInterest"])[["date", "openInterest"]].sort_values("date").reset_index(drop=True)


def fetch_long_short_ratio(symbol: str = "BTCUSDT", period: str = "1d", limit: int = 200) -> pd.DataFrame:
    """Fetch top trader long/short ratio from Binance futures.

    Args:
        symbol (str): Symbol.
        period (str): '5m','15m','1h','4h','1d'.
        limit (int): Max records.

    Returns:
        pd.DataFrame: ['date','longShortRatio']
    """

    q = urlencode({"symbol": symbol, "period": period, "limit": limit})
    url = f"{_BINANCE_FAPI}/futures/data/topLongShortAccountRatio?{q}"
    data = _get_json(url)
    if not isinstance(data, list):
        return pd.DataFrame(columns=["date", "longShortRatio"])
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["longShortRatio"] = pd.to_numeric(df["longShortRatio"], errors="coerce")
    return df.dropna(subset=["date", "longShortRatio"])[["date", "longShortRatio"]].sort_values("date").reset_index(drop=True)


def fetch_options_iv_placeholder() -> pd.DataFrame:
    """Placeholder for options implied volatility.

    Returns:
        pd.DataFrame: Empty DataFrame with ['date','iv'].
    """

    return pd.DataFrame(columns=["date", "iv"])
