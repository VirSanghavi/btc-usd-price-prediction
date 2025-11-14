"""Sentiment data sources and placeholders."""
from __future__ import annotations

import json
from typing import Optional
from urllib.request import Request, urlopen

import pandas as pd


def fetch_fear_greed_index() -> pd.DataFrame:
    """Fetch Crypto Fear & Greed Index from alternative.me.

    Returns:
        pd.DataFrame: Columns ['date','value','classification'].
    """

    try:
        url = "https://api.alternative.me/fng/?limit=0&format=json"
        req = Request(url, headers={"User-Agent": "crypto-quant/1.0"})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        values = data.get("data", [])
        df = pd.DataFrame(values)
        df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["classification"] = df.get("value_classification", "")
        return df.dropna(subset=["date", "value"])[["date", "value", "classification"]].sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["date", "value", "classification"])


def reddit_sentiment_placeholder() -> pd.DataFrame:
    """Placeholder for Reddit sentiment (to be implemented if API access available)."""

    return pd.DataFrame(columns=["date", "score"])


def news_sentiment_placeholder() -> pd.DataFrame:
    """Placeholder for News sentiment (to be implemented with a news API)."""

    return pd.DataFrame(columns=["date", "score"])
