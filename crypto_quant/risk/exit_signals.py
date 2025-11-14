"""Risk-adjusted exit signals combining multiple conditions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators.technical import rsi, macd


def exit_signal(df: pd.DataFrame, regime: str = "unknown", funding_rate: float | None = None) -> dict:
    """Compute a composite exit signal and human-readable explanation.

    Conditions considered:
      - RSI overbought zones
      - Volatility spikes (proxy: recent std of returns)
      - Regime switching (bear regime)
      - Funding rate flips (negative)
      - MACD bear crossover
      - Price below MA200

    Args:
        df (pd.DataFrame): Data with 'close' or 'price' and 'high','low' optionally.
        regime (str): Current regime label from HMM.
        funding_rate (float | None): Latest funding rate.

    Returns:
        dict: {'signal': bool, 'score': float, 'explanation': str}
    """

    data = df.copy()
    if "close" not in data.columns and "price" in data.columns:
        data["close"] = data["price"]
    close = data["close"].dropna()
    score = 0.0
    parts: list[str] = []

    # RSI overbought
    rsi_val = float(rsi(close).iloc[-1]) if len(close) > 20 else np.nan
    if not np.isnan(rsi_val) and rsi_val > 70:
        score += 1.0
        parts.append(f"RSI is overbought at {rsi_val:.1f} (>70)")

    # Volatility spike (recent std > 75th percentile)
    rets = np.log(close).diff()
    recent_vol = float(rets.tail(20).std()) if len(rets) > 20 else np.nan
    vol_thresh = float(rets.rolling(60).std().dropna().quantile(0.75)) if len(rets) > 60 else np.nan
    if not np.isnan(recent_vol) and not np.isnan(vol_thresh) and recent_vol > vol_thresh:
        score += 0.8
        parts.append("Recent volatility spike above 75th percentile")

    # Regime check
    if regime == "bear":
        score += 1.2
        parts.append("HMM indicates bear regime")

    # Funding rate flip (negative)
    if funding_rate is not None and funding_rate < 0:
        score += 0.5
        parts.append("Funding rate negative (short bias)")

    # MACD bear crossover
    macd_df = macd(close)
    if len(macd_df) > 2:
        if macd_df["macd"].iloc[-1] < macd_df["signal"].iloc[-1] and macd_df["macd"].iloc[-2] >= macd_df["signal"].iloc[-2]:
            score += 1.0
            parts.append("MACD bear crossover detected")

    # Price below MA200
    ma200 = close.rolling(200).mean()
    if len(ma200.dropna()) > 0 and close.iloc[-1] < ma200.iloc[-1]:
        score += 0.7
        parts.append("Price below 200-day moving average")

    signal = score >= 2.0
    if not parts:
        parts.append("No major risk conditions triggered")
    explanation = "; ".join(parts)
    return {"signal": signal, "score": score, "explanation": explanation}
