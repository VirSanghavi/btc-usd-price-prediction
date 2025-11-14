"""Volatility models and estimators."""
from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(returns: pd.Series, window: int = 30, trading_days: int = 365) -> pd.Series:
    """Compute annualized realized volatility from returns.

    Args:
        returns (pd.Series): Log or arithmetic daily returns.
        window (int): Rolling window length.
        trading_days (int): Annualization factor.

    Returns:
        pd.Series: Annualized volatility.
    """

    vol = returns.rolling(window).std() * np.sqrt(trading_days)
    return vol


def garch_forecast(returns: pd.Series, horizon: int = 1) -> dict:
    """Fit a GARCH(1,1) model and forecast variance.

    Args:
        returns (pd.Series): Daily returns.
        horizon (int): Steps ahead to forecast.

    Returns:
        dict: {'model': fitted, 'forecast_var': np.ndarray}
    """

    try:
        from arch import arch_model  # type: ignore

        r = returns.dropna() * 100  # scale
        if len(r) < 50:
            return {"model": None, "forecast_var": np.array([])}
        am = arch_model(r, vol='Garch', p=1, q=1, dist='normal', mean='zero')
        res = am.fit(disp='off')
        f = res.forecast(horizon=horizon)
        var = f.variance.values[-1]
        return {"model": res, "forecast_var": var}
    except Exception:
        return {"model": None, "forecast_var": np.array([])}


def egarch_forecast(returns: pd.Series, horizon: int = 1) -> dict:
    """Fit an EGARCH(1,1) model and forecast variance.

    Args:
        returns (pd.Series): Daily returns.
        horizon (int): Steps ahead.

    Returns:
        dict: {'model': fitted, 'forecast_var': np.ndarray}
    """

    try:
        from arch import arch_model  # type: ignore

        r = returns.dropna() * 100
        if len(r) < 50:
            return {"model": None, "forecast_var": np.array([])}
        am = arch_model(r, vol='EGARCH', p=1, o=0, q=1, dist='normal', mean='zero')
        res = am.fit(disp='off')
        f = res.forecast(horizon=horizon)
        var = f.variance.values[-1]
        return {"model": res, "forecast_var": var}
    except Exception:
        return {"model": None, "forecast_var": np.array([])}


def intraday_volatility_placeholder() -> pd.DataFrame:
    """Placeholder for intraday volatility estimation.

    Returns:
        pd.DataFrame: Empty structure.
    """

    return pd.DataFrame(columns=["timestamp", "volatility"])
