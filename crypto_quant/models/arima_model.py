"""ARIMA modeling via pmdarima."""
from __future__ import annotations

import pandas as pd


def fit_arima_forecast(df: pd.DataFrame, horizons=(1, 7, 30)) -> dict:
    """Fit ARIMA model on daily close and forecast given horizons.

    Args:
        df (pd.DataFrame): Daily data with 'date' and 'close' or 'price'.
        horizons (tuple): Forecast horizons in days.

    Returns:
        dict: {horizon: forecast_value}
    """

    try:
        import pmdarima as pm  # type: ignore

        data = df.copy()
        if "close" not in data.columns and "price" in data.columns:
            data["close"] = data["price"]
        y = data["close"].dropna().astype(float).values
        if len(y) < 30:
            return {h: float("nan") for h in horizons}
        model = pm.auto_arima(y, seasonal=False, error_action="ignore", suppress_warnings=True)
        out = {}
        for h in horizons:
            fc = model.predict(n_periods=h)
            out[h] = float(fc[-1])
        return out
    except Exception:
        return {h: float("nan") for h in horizons}
