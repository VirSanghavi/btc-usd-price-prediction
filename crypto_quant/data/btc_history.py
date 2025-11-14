"""BTC historical data loader with Kaggle and CSV/yfinance fallback."""
from __future__ import annotations

import os
import io
import zipfile
import datetime as dt
from typing import Optional
from urllib.request import urlopen

import numpy as np
import pandas as pd


def _try_kaggle_daily() -> Optional[pd.DataFrame]:
    """Attempt to load a daily BTC dataset via kagglehub.

    Returns:
        Optional[pd.DataFrame]: DataFrame if available, else None.
    """

    try:
        import kagglehub  # type: ignore

        # Use a lightweight daily dataset if available
        # Example: 'mczielinski/bitcoin-historical-data' contains minute bars; heavy to load.
        # Instead, try a daily dataset. If not found, fall through to fallback.
        # We'll try CoinDesk daily OHLC (often mirrored), else skip.
        # Note: kagglehub.fetch returns a local path after download.
        candidates = [
            # (dataset, relative path inside archive)
            ("mczielinski/bitcoin-historical-data", "coinbaseUSD_1-dag_data.csv"),
        ]
        for ds, rel in candidates:
            try:
                local_dir = kagglehub.dataset_download(ds)
                candidate_path = os.path.join(local_dir, rel)
                if os.path.exists(candidate_path):
                    df = pd.read_csv(candidate_path)
                    return _normalize_daily(df)
            except Exception:
                continue
    except Exception:
        pass
    return None


def _normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize daily OHLC dataset to standard columns.

    Args:
        df (pd.DataFrame): Raw DataFrame.

    Returns:
        pd.DataFrame: Cleaned with columns [date, open, high, low, close, volume, price].
    """

    df = df.copy()
    # Try common column names
    rename_map = {
        "Date": "date",
        "date": "date",
        "Timestamp": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Volume USD": "volume",
    }
    df = df.rename(columns=rename_map)
    if "date" not in df.columns:
        # attempt converting UNIX timestamp column
        for c in df.columns:
            if c.lower() in ("timestamp", "time"):
                df["date"] = pd.to_datetime(df[c], unit="s", utc=True)
                break
    if "date" not in df.columns:
        raise ValueError("No date column found after normalization")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # price alias
    if "close" in df.columns:
        df["price"] = df["close"]
    elif "open" in df.columns:
        df["price"] = df["open"]
    else:
        # if only a Price column exists
        for c in df.columns:
            if c.lower() == "price":
                df["price"] = pd.to_numeric(df[c], errors="coerce")
                break
    df = df.sort_values("date").dropna(subset=["date", "price"]).reset_index(drop=True)
    return df[[c for c in ["date", "open", "high", "low", "close", "volume", "price"] if c in df.columns]]


def _fallback_yfinance(period: str = "max") -> pd.DataFrame:
    """Fallback to yfinance BTC-USD daily history.

    Args:
        period (str): yfinance period, default 'max'.

    Returns:
        pd.DataFrame: normalized daily data.
    """

    import yfinance as yf  # lazy import

    t = yf.Ticker("BTC-USD")
    hist = t.history(period=period, interval="1d", auto_adjust=False)
    hist = hist.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    hist["date"] = pd.to_datetime(hist["date"], utc=True)
    hist["price"] = hist["close"]
    return hist[["date", "open", "high", "low", "close", "volume", "price"]]


def load_btc_history() -> pd.DataFrame:
    """Load full BTC historical daily dataset.

    Tries Kaggle via kagglehub, then CSV path set in BTC_HIST_CSV env var, then yfinance.

    Returns:
        pd.DataFrame: Daily BTC data with 'date' and 'price' at minimum.
    """

    # 1) Environment CSV path
    csv_path = os.getenv("BTC_HIST_CSV")
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return _normalize_daily(df)
        except Exception:
            pass

    # 2) Kaggle attempt
    kaggle_df = _try_kaggle_daily()
    if kaggle_df is not None:
        return kaggle_df

    # 3) yfinance fallback
    return _fallback_yfinance()


def last_3_months(df: pd.DataFrame) -> pd.DataFrame:
    """Return last ~3 months of data from a daily DataFrame.

    Args:
        df (pd.DataFrame): Full daily data with 'date'.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """

    if df.empty:
        return df
    end = df["date"].max()
    start = end - pd.Timedelta(days=92)
    return df[df["date"].between(start, end)].reset_index(drop=True)
