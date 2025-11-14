"""Monte Carlo simulation for BTC hitting a target price."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mc_hit_probability(df: pd.DataFrame, target_price: float = 99_000.0, paths: int = 10_000, horizons=(1, 7, 30)) -> dict:
    """Run geometric Brownian motion simulations to estimate hit probability.

    Args:
        df (pd.DataFrame): Data with 'close' or 'price'.
        target_price (float): Threshold to hit.
        paths (int): Number of simulated paths.
        horizons (tuple): Days to evaluate hit probability within.

    Returns:
        dict: {horizon: probability}
    """

    data = df.copy()
    if "close" not in data.columns and "price" in data.columns:
        data["close"] = data["price"]
    px = data["close"].dropna().astype(float).values
    if len(px) < 50:
        return {h: float("nan") for h in horizons}
    log_ret = np.diff(np.log(px))
    mu = float(np.mean(log_ret))
    sigma = float(np.std(log_ret))
    s0 = float(px[-1])

    out: dict[int, float] = {}
    for H in horizons:
        dt = 1.0
        steps = int(H)
        # simulate paths
        Z = np.random.normal(size=(paths, steps))
        increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        log_paths = np.cumsum(increments, axis=1)
        S = s0 * np.exp(log_paths)
        hit = (S >= target_price).any(axis=1)
        out[H] = float(hit.mean())
    return out
