"""Macro data such as DXY, SP500 correlation, and US interest rates."""
from __future__ import annotations

import json
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from ..utils.config import get_api_keys


def fetch_dxy_yfinance(period: str = "5y") -> pd.DataFrame:
    """Fetch DXY index via yfinance.

    Args:
        period (str): Period string for yfinance, default '5y'.

    Returns:
        pd.DataFrame: ['date','close'] series.
    """

    try:
        import yfinance as yf

        t = yf.Ticker("DX-Y.NYB")  # DXY index
        hist = t.history(period=period, interval="1d")
        hist = hist.reset_index()[["Date", "Close"]]
        hist = hist.rename(columns={"Date": "date", "Close": "close"})
        hist["date"] = pd.to_datetime(hist["date"], utc=True)
        return hist.dropna().reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["date", "close"])


def sp500_correlation_with_btc(btc_df: pd.DataFrame, lookback: int = 90) -> float:
    """Compute correlation between BTC daily returns and S&P 500 daily returns.

    Args:
        btc_df (pd.DataFrame): BTC daily data with 'date' and 'close' or 'price'.
        lookback (int): Days of lookback for correlation.

    Returns:
        float: Pearson correlation, np.nan if unavailable.
    """

    try:
        import yfinance as yf

        sp = yf.Ticker("^GSPC")
        sp_hist = sp.history(period="2y", interval="1d").reset_index()
        sp_hist = sp_hist.rename(columns={"Date": "date", "Close": "sp_close"})
        sp_hist["date"] = pd.to_datetime(sp_hist["date"], utc=True)
        b = btc_df.copy()
        if "close" not in b.columns and "price" in b.columns:
            b["close"] = b["price"]
        b = b[["date", "close"]]
        merged = pd.merge_asof(
            b.sort_values("date"), sp_hist.sort_values("date"), on="date"
        ).dropna()
        merged["btc_ret"] = merged["close"].pct_change()
        merged["sp_ret"] = merged["sp_close"].pct_change()
        tail = merged.tail(lookback)
        return float(tail[["btc_ret", "sp_ret"]].corr().iloc[0, 1])
    except Exception:
        return float("nan")


def fetch_fred_interest_rates() -> pd.DataFrame:
    """Fetch US interest rates (2Y, 10Y) from FRED if API key exists.

    Returns:
        pd.DataFrame: Columns ['date','DGS2','DGS10'] if successful, else empty.
    """

    keys = get_api_keys()
    if not keys.fred:
        return pd.DataFrame(columns=["date", "DGS2", "DGS10"])
    try:
        series = {"DGS2": "DGS2", "DGS10": "DGS10"}
        frames = []
        for name, sid in series.items():
            q = urlencode(
                {
                    "series_id": sid,
                    "api_key": keys.fred,
                    "file_type": "json",
                    "observation_start": "2000-01-01",
                }
            )
            url = f"https://api.stlouisfed.org/fred/series/observations?{q}"
            req = Request(url, headers={"User-Agent": "crypto-quant/1.0"})
            with urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            obs = pd.DataFrame(data.get("observations", []))[["date", "value"]]
            obs["date"] = pd.to_datetime(obs["date"], utc=True)
            obs[name] = pd.to_numeric(obs["value"], errors="coerce")
            frames.append(obs[["date", name]])
        out = frames[0]
        for f in frames[1:]:
            out = pd.merge(out, f, on="date", how="outer")
        out = out.sort_values("date").reset_index(drop=True)
        return out
    except Exception:
        return pd.DataFrame(columns=["date", "DGS2", "DGS10"])
