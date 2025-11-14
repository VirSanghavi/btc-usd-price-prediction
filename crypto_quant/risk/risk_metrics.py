"""Risk metrics: Sharpe, Sortino, Max Drawdown, Calmar, Rolling metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 365) -> float:
    """Compute annualized Sharpe ratio.

    Args:
        returns (pd.Series): Periodic returns.
        risk_free (float): Risk-free rate per period.
        periods_per_year (int): Annualization factor.

    Returns:
        float: Sharpe ratio.
    """

    excess = returns - risk_free
    if excess.std(ddof=0) == 0 or excess.dropna().empty:
        return float("nan")
    return float((excess.mean() / excess.std(ddof=0)) * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 365) -> float:
    """Compute annualized Sortino ratio (downside deviation)."""

    excess = returns - risk_free
    downside = excess.copy()
    downside[downside > 0] = 0
    dd = downside.std(ddof=0)
    if dd == 0 or excess.dropna().empty:
        return float("nan")
    return float((excess.mean() / dd) * np.sqrt(periods_per_year))


def max_drawdown(prices: pd.Series) -> float:
    """Maximum drawdown as min of (price/peak - 1)."""

    if prices.dropna().empty:
        return float("nan")
    cummax = prices.cummax()
    drawdown = prices / cummax - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, prices: pd.Series, periods_per_year: int = 365) -> float:
    """Calmar ratio = annualized return / |max drawdown|."""

    ann_ret = returns.mean() * periods_per_year
    mdd = abs(max_drawdown(prices))
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(ann_ret / mdd)


def rolling_risk_metrics(returns: pd.Series, window: int = 30) -> pd.DataFrame:
    """Rolling Sharpe and volatility."""

    roll_vol = returns.rolling(window).std() * np.sqrt(365)
    roll_mean = returns.rolling(window).mean()
    roll_sharpe = (roll_mean / returns.rolling(window).std()) * np.sqrt(365)
    return pd.DataFrame({"roll_vol": roll_vol, "roll_sharpe": roll_sharpe})
