"""ARCH/GARCH volatility modeling using arch library."""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_garch_and_forecast(returns: pd.Series, horizon: int = 5) -> dict:
    """Fit a GARCH(1,1) model and return volatility forecasts.

    Args:
        returns (pd.Series): Daily returns series.
        horizon (int): Steps ahead for forecast.

    Returns:
        dict: {'model': fitted_or_None, 'vol_forecast': np.ndarray}
    """

    try:
        from arch import arch_model  # type: ignore

        r = (returns.dropna() * 100).astype(float)
        if len(r) < 50:
            return {"model": None, "vol_forecast": np.array([])}
        am = arch_model(r, vol='Garch', p=1, q=1, dist='normal', mean='zero')
        res = am.fit(disp='off')
        fore = res.forecast(horizon=horizon)
        var = fore.variance.values[-1]
        vol = np.sqrt(var)  # percent terms
        return {"model": res, "vol_forecast": vol}
    except Exception:
        return {"model": None, "vol_forecast": np.array([])}
