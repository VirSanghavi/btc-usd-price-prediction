"""Prophet forecasting model."""
from __future__ import annotations

import pandas as pd


def fit_prophet_forecast(df: pd.DataFrame, horizons=(1, 7, 30)) -> dict:
    """Fit Prophet on daily close and forecast horizons.

    Args:
        df (pd.DataFrame): DataFrame with 'date' and 'close' or 'price'.
        horizons (tuple): Days ahead to forecast.

    Returns:
        dict: {horizon: forecast_value}
    """

    try:
        from prophet import Prophet  # type: ignore

        data = df.copy()
        if "close" not in data.columns and "price" in data.columns:
            data["close"] = data["price"]
        d = data[["date", "close"]].dropna().rename(columns={"date": "ds", "close": "y"})
        if len(d) < 30:
            return {h: float("nan") for h in horizons}
        m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
        m.fit(d)
        out = {}
        max_h = max(horizons)
        future = m.make_future_dataframe(periods=max_h, freq="D")
        fc = m.predict(future)
        fc = fc.set_index("ds")
        for h in horizons:
            ts = d["ds"].max() + pd.Timedelta(days=h)
            if ts in fc.index:
                out[h] = float(fc.loc[ts, "yhat"])  # type: ignore
            else:
                out[h] = float("nan")
        return out
    except Exception:
        return {h: float("nan") for h in horizons}
