"""Hidden Markov Model for market regimes using hmmlearn."""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_hmm_regimes(df: pd.DataFrame, n_states: int = 3) -> dict:
    """Fit an HMM on daily returns to detect bull/bear/sideways regimes.

    Args:
        df (pd.DataFrame): Data with 'date' and 'close' or 'price'.
        n_states (int): Number of regimes.

    Returns:
        dict: {'model': model_or_None, 'states': pd.Series, 'current_regime': str, 'probs': np.ndarray}
    """

    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore

        data = df.copy()
        if "close" not in data.columns and "price" in data.columns:
            data["close"] = data["price"]
        data = data.dropna(subset=["close"]).copy()
        data["ret"] = np.log(data["close"]).diff()
        X = data["ret"].dropna().values.reshape(-1, 1)
        if len(X) < 50:
            return {"model": None, "states": pd.Series(dtype=int), "current_regime": "unknown", "probs": np.array([])}
        hmm = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200)
        hmm.fit(X)
        states = hmm.predict(X)
        means = np.array([X[states == i].mean() for i in range(n_states)])
        order = np.argsort(means)
        mapping = {order[0]: "bear", order[1]: "sideways", order[2]: "bull"}
        labeled_states = pd.Series([mapping[s] for s in states], index=data.index[data.index.get_loc(data.index.min()) + 1 :])
        # Compute posterior probs for last point
        logprob, posteriors = hmm.score_samples(X)
        last_post = posteriors[-1]
        current_regime = mapping[np.argmax(last_post)]
        return {"model": hmm, "states": labeled_states, "current_regime": current_regime, "probs": last_post}
    except Exception:
        return {"model": None, "states": pd.Series(dtype=int), "current_regime": "unknown", "probs": np.array([])}
