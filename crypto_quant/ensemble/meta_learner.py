"""Meta-learner to blend forecasts across models with regime awareness."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _normalize_forecasts(forecasts: Dict[str, Dict[int, float]], last_price: float) -> pd.DataFrame:
    """Convert a forecast dict-of-dicts into a DataFrame indexed by horizon.

    Args:
        forecasts: {'model': {horizon_days: price}}
        last_price: Last observed price (used for 1h scaling and fallbacks).

    Returns:
        pd.DataFrame: columns per model, index=horizon days.
    """

    horizons = sorted({h for d in forecasts.values() for h in d.keys()})
    out = pd.DataFrame(index=horizons)
    for model, d in forecasts.items():
        out[model] = pd.Series(d)
    # Fill missing with last_price to avoid NaNs skewing weights
    return out.fillna(last_price)


def _regime_weights(regime: str) -> Dict[str, float]:
    """Assign heuristic weights per regime when backtest not available."""

    if regime == "bull":
        return {"arima": 0.25, "prophet": 0.35, "lstm": 0.25, "hmm": 0.05, "mc": 0.10}
    if regime == "bear":
        return {"arima": 0.25, "prophet": 0.20, "lstm": 0.15, "hmm": 0.20, "mc": 0.20}
    # sideways
    return {"arima": 0.30, "prophet": 0.25, "lstm": 0.20, "hmm": 0.10, "mc": 0.15}


def blend_forecasts(
    forecasts: Dict[str, Dict[int, float]],
    last_price: float,
    regime: str = "unknown",
    backtest_errors: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Blend model forecasts using inverse-error weighting and regime heuristics.

    Args:
        forecasts: {'model': {horizon_days: price}}
        last_price: Last observed price.
        regime: Current regime.
        backtest_errors: Optional {'model': MAE} from recent backtest.

    Returns:
        dict: {'1h': price, '1d': price, '1w': price, '1m': price}
    """

    df = _normalize_forecasts(forecasts, last_price)

    # Determine base weights
    if backtest_errors and len(backtest_errors) > 0:
        # inverse MAE weighting
        inv = {m: 1.0 / e if e and np.isfinite(e) and e > 0 else 0.0 for m, e in backtest_errors.items()}
        tot = sum(inv.values()) or 1.0
        w = {m: v / tot for m, v in inv.items()}
        # If some models missing in errors, distribute small uniform weight
        for m in df.columns:
            if m not in w:
                w[m] = 0.0
    else:
        w = _regime_weights(regime)
        # ensure all columns present
        for m in df.columns:
            w.setdefault(m, 0.0)
        # normalize
        s = sum(w.values()) or 1.0
        w = {k: v / s for k, v in w.items()}

    w_vec = np.array([w.get(c, 0.0) for c in df.columns])
    blended_by_day = df.values @ w_vec

    # Map horizons to required outputs
    result: Dict[str, float] = {}
    # 1d
    if 1 in df.index:
        result["1d"] = float(blended_by_day[list(df.index).index(1)])
    else:
        result["1d"] = float(last_price)
    # 1w (7d)
    if 7 in df.index:
        result["1w"] = float(blended_by_day[list(df.index).index(7)])
    else:
        result["1w"] = float(result["1d"])  # fallback
    # 1m (30d)
    if 30 in df.index:
        result["1m"] = float(blended_by_day[list(df.index).index(30)])
    else:
        result["1m"] = float(result["1w"])
    # 1h: scale from 1d assuming 24h scaling of expected change
    exp_change_1d = result["1d"] / last_price - 1.0 if last_price else 0.0
    result["1h"] = float(last_price * (1 + exp_change_1d / 24.0))

    return result
