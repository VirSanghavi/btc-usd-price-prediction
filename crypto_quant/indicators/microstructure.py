"""Microstructure placeholders for bid-ask spread, order book imbalance, and more.

These require high-frequency data feeds and are implemented as placeholders.
"""
from __future__ import annotations

import pandas as pd


def bid_ask_spread_placeholder() -> pd.DataFrame:
    """Placeholder for bid-ask spread time series."""
    return pd.DataFrame(columns=["timestamp", "spread"])


def orderbook_imbalance_placeholder() -> pd.DataFrame:
    """Placeholder for order book imbalance."""
    return pd.DataFrame(columns=["timestamp", "imbalance"])


def market_depth_slope_placeholder() -> pd.DataFrame:
    """Placeholder for market depth slope."""
    return pd.DataFrame(columns=["timestamp", "slope"])


def hf_skew_kurtosis_placeholder() -> pd.DataFrame:
    """Placeholder for high-frequency skewness and kurtosis metrics."""
    return pd.DataFrame(columns=["timestamp", "skew", "kurtosis"])
