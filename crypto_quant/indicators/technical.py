"""Technical indicators for BTC time series."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average.

    Args:
        series (pd.Series): Input series.
        span (int): EMA span.

    Returns:
        pd.Series: EMA.
    """

    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index.

    Args:
        series (pd.Series): Price series.
        period (int): Lookback period.

    Returns:
        pd.Series: RSI values (0-100).
    """

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50.0)


def multi_timeframe_rsi(series: pd.Series, periods: list[int] = [14, 30, 50]) -> pd.DataFrame:
    """Compute RSI for multiple lookbacks.

    Args:
        series (pd.Series): Price series.
        periods (list[int]): List of RSI periods.

    Returns:
        pd.DataFrame: Columns named 'rsi_{period}'.
    """

    out = pd.DataFrame(index=series.index)
    for p in periods:
        out[f"rsi_{p}"] = rsi(series, p)
    return out


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal, and histogram.

    Args:
        series (pd.Series): Price series.
        fast (int): Fast EMA span.
        slow (int): Slow EMA span.
        signal (int): Signal EMA span.

    Returns:
        pd.DataFrame: Columns ['macd','signal','hist'].
    """

    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator (%K, %D).

    Args:
        high (pd.Series): High prices.
        low (pd.Series): Low prices.
        close (pd.Series): Close prices.
        k_period (int): %K lookback.
        d_period (int): %D smoothing of %K.

    Returns:
        pd.DataFrame: Columns ['stoch_k','stoch_d'].
    """

    lowest_low = low.rolling(k_period, min_periods=k_period).min()
    highest_high = high.rolling(k_period, min_periods=k_period).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(d_period, min_periods=d_period).mean()
    return pd.DataFrame({"stoch_k": stoch_k, "stoch_d": stoch_d})


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands with BandWidth.

    Args:
        series (pd.Series): Price series.
        period (int): SMA period.
        num_std (float): Standard deviations.

    Returns:
        pd.DataFrame: ['bb_mid','bb_upper','bb_lower','bb_bandwidth'].
    """

    mid = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = (upper - lower) / mid
    return pd.DataFrame({
        "bb_mid": mid,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_bandwidth": bandwidth,
    })


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range.

    Args:
        high (pd.Series), low (pd.Series), close (pd.Series)
        period (int): Lookback.

    Returns:
        pd.Series: ATR values.
    """

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change indicator.

    Args:
        series (pd.Series): Price series.
        period (int): Lookback period.

    Returns:
        pd.Series: ROC values.
    """

    return series.pct_change(periods=period) * 100


def vwap_deviation(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Deviation of price from VWAP.

    Returns:
        pd.Series: (close - vwap) / vwap
    """

    typical = (high + low + close) / 3.0
    cum_vp = (typical * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    vwap = cum_vp / cum_vol
    return (close - vwap) / vwap


def fractal_dimension_index(series: pd.Series, window: int = 14) -> pd.Series:
    """Fractal Dimension Index (FDI) estimator.

    Uses the approach based on the ratio of multi-scale path lengths.

    Args:
        series (pd.Series): Price series.
        window (int): Rolling window size.

    Returns:
        pd.Series: FDI values roughly in [1,2]. Higher implies more noise.
    """

    s = series.apply(np.log).replace([np.inf, -np.inf], np.nan)
    def _fdi(x: pd.Series) -> float:
        x = x.dropna().values
        if len(x) < 5:
            return np.nan
        L1 = np.sum(np.abs(np.diff(x)))
        L2 = np.sum(np.abs(x[2:] - x[:-2])) * (len(x) - 1) / (len(x) - 2) if len(x) > 2 else np.nan
        if L1 <= 0 or L2 <= 0 or np.isnan(L2):
            return np.nan
        return 1 + np.log(L2 / L1) / np.log(2)
    return s.rolling(window).apply(_fdi, raw=False)


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """Estimate Hurst exponent using aggregated variance method.

    Args:
        series (pd.Series): Price series.
        max_lag (int): Maximum lag to consider.

    Returns:
        float: H in (0,1). 0.5 ~ random walk, >0.5 trending, <0.5 mean-reverting.
    """

    x = np.log(series.replace(0, np.nan)).dropna().values
    if len(x) < max_lag * 2:
        return np.nan
    taus = np.arange(2, max_lag + 1)
    variances = [np.var(x[lag:] - x[:-lag]) for lag in taus]
    with np.errstate(divide='ignore', invalid='ignore'):
        poly = np.polyfit(np.log(taus), np.log(variances), 1)
    hurst = poly[0] / 2.0
    return float(hurst)
