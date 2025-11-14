"""On-chain metrics via Glassnode where possible."""
from __future__ import annotations

import json
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from ..utils.config import get_api_keys


def _glassnode_fetch(metric: str, params: dict) -> Optional[pd.DataFrame]:
    """Fetch a metric from Glassnode API if key present.

    Args:
        metric (str): Metric endpoint slug (e.g., 'mvrv_z_score').
        params (dict): Query params including 'a' (asset) and 'i' (interval).

    Returns:
        Optional[pd.DataFrame]: DataFrame with 'date' and 'value' or None.
    """

    keys = get_api_keys()
    if not keys.glassnode:
        return None
    try:
        base = f"https://api.glassnode.com/v1/metrics/indicators/{metric}"
        q = params.copy()
        q["api_key"] = keys.glassnode
        url = f"{base}?{urlencode(q)}"
        req = Request(url, headers={"User-Agent": "crypto-quant/1.0"})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        df = pd.DataFrame(data)
        if "t" in df.columns:
            df["date"] = pd.to_datetime(df["t"], unit="s", utc=True)
        elif "time" in df.columns:
            df["date"] = pd.to_datetime(df["time"], unit="s", utc=True)
        else:
            # Some endpoints may deliver 'timestamp'
            for c in df.columns:
                if c.lower() in ("timestamp", "ts"):
                    df["date"] = pd.to_datetime(df[c], unit="s", utc=True)
                    break
        if "v" in df.columns:
            df["value"] = pd.to_numeric(df["v"], errors="coerce")
        elif "value" not in df.columns and "o" in df.columns:
            df["value"] = pd.to_numeric(df["o"], errors="coerce")
        df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
        return df[["date", "value"]]
    except Exception:
        return None


def fetch_mvrv_z_score() -> pd.DataFrame:
    """Fetch Glassnode MVRV Z-score for BTC if possible.

    Returns:
        pd.DataFrame: Columns ['date','value'] or empty DataFrame.
    """

    df = _glassnode_fetch("mvrv_z_score", {"a": "BTC", "i": "24h"})
    return df if df is not None else pd.DataFrame(columns=["date", "value"])


def fetch_nvt_ratio() -> pd.DataFrame:
    """Fetch Glassnode NVT ratio for BTC if possible.

    Returns:
        pd.DataFrame: Columns ['date','value'] or empty DataFrame.
    """

    df = _glassnode_fetch("nvt", {"a": "BTC", "i": "24h"})
    return df if df is not None else pd.DataFrame(columns=["date", "value"])
