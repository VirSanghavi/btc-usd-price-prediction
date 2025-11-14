"""Main pipeline orchestrating data, indicators, models, ensemble, and output."""
from __future__ import annotations

import math
import os
from typing import Dict

import numpy as np
import pandas as pd

from .utils.config import set_seed
from .data.btc_history import load_btc_history
from .data.realtime import fetch_btc_price_coingecko
from .data.onchain import fetch_mvrv_z_score, fetch_nvt_ratio
from .data.derivatives import fetch_funding_rate, fetch_open_interest, fetch_long_short_ratio
from .data.sentiment import fetch_fear_greed_index
from .data.macro import fetch_dxy_yfinance, sp500_correlation_with_btc, fetch_fred_interest_rates
from .indicators.technical import multi_timeframe_rsi, macd, stochastic_oscillator, bollinger_bands, atr, roc, vwap_deviation, fractal_dimension_index, hurst_exponent
from .indicators.volatility import realized_volatility
from .models.arima_model import fit_arima_forecast
from .models.prophet_model import fit_prophet_forecast
from .models.lstm_model import lstm_predict_next_prices
from .models.hmm_regime import fit_hmm_regimes
from .models.monte_carlo import mc_hit_probability
from .models.garch_model import fit_garch_and_forecast
from .risk.risk_metrics import sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, rolling_risk_metrics
from .risk.exit_signals import exit_signal
from .ensemble.meta_learner import blend_forecasts


def _ensure_close(df: pd.DataFrame) -> pd.DataFrame:
    if "close" not in df.columns and "price" in df.columns:
        df = df.copy()
        df["close"] = df["price"]
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a suite of technical indicators and return merged DataFrame."""

    d = _ensure_close(df)
    out = d.copy()
    out = out.sort_values("date").reset_index(drop=True)
    close = out["close"]
    high = out.get("high", close)
    low = out.get("low", close)
    volume = out.get("volume", pd.Series(index=out.index, data=np.nan)).fillna(method="ffill").fillna(0)

    out = pd.concat([
        out,
        multi_timeframe_rsi(close),
        macd(close),
        stochastic_oscillator(high, low, close),
        bollinger_bands(close),
    ], axis=1)
    out["atr"] = atr(high, low, close)
    out["roc_12"] = roc(close)
    out["vwap_dev"] = vwap_deviation(high, low, close, volume)
    out["fdi_14"] = fractal_dimension_index(close)
    out["hurst"] = hurst_exponent(close)
    out["ret"] = np.log(close).diff()
    out["realized_vol_30"] = realized_volatility(out["ret"])
    return out


def compute_risk_metrics(out: pd.DataFrame) -> Dict[str, float]:
    close = out["close"].dropna()
    rets = np.log(close).diff().dropna()
    return {
        "sharpe": sharpe_ratio(rets),
        "sortino": sortino_ratio(rets),
        "max_drawdown": max_drawdown(close),
        "calmar": calmar_ratio(rets, close),
    }


def run_models(out: pd.DataFrame) -> Dict[str, dict]:
    """Run all models and collect their outputs."""

    arima_fc = fit_arima_forecast(out)
    prophet_fc = fit_prophet_forecast(out)
    lstm_fc = lstm_predict_next_prices(out)
    hmm = fit_hmm_regimes(out)
    mc = mc_hit_probability(out)
    garch = fit_garch_and_forecast(out["ret"]) if "ret" in out.columns else {"model": None, "vol_forecast": np.array([])}

    models = {
        "arima": arima_fc,
        "prophet": prophet_fc,
        "lstm": lstm_fc,
        "hmm": hmm,
        "mc": mc,
        "garch": garch,
    }
    return models


def main() -> None:
    set_seed(42)

    # 1. Load historical data
    hist = load_btc_history()
    hist = _ensure_close(hist)

    # 2. Fetch realtime price
    rt_price = fetch_btc_price_coingecko()
    if rt_price and not math.isnan(rt_price):
        # append latest point as today if newer
        last_date = hist["date"].max().normalize()
        today = pd.Timestamp.utcnow().normalize()
        if today > last_date:
            row = {"date": today, "close": float(rt_price), "price": float(rt_price)}
            hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)

    # 3. Compute indicators and risk metrics
    feats = compute_indicators(hist)
    risk = compute_risk_metrics(feats)

    # ancillary data (best-effort)
    mvrv = fetch_mvrv_z_score()
    nvt = fetch_nvt_ratio()
    funding = fetch_funding_rate().tail(1)
    oi = fetch_open_interest().tail(1)
    lsr = fetch_long_short_ratio().tail(1)
    fng = fetch_fear_greed_index().tail(1)
    dxy = fetch_dxy_yfinance().tail(1)
    spcorr = sp500_correlation_with_btc(hist)
    rates = fetch_fred_interest_rates().tail(1)

    # 4. Run ALL models
    models = run_models(feats)

    # 5. Ensemble meta-learner
    last_price = float(feats["close"].iloc[-1])
    arima_fc = models["arima"]
    prophet_fc = models["prophet"]
    lstm_next_price = models["lstm"].get("next_price", float("nan"))
    forecasts = {
        "arima": arima_fc,
        "prophet": prophet_fc,
        # map LSTM next to 1-day horizon
        "lstm": {1: float(lstm_next_price) if np.isfinite(lstm_next_price) else float("nan")},
        # HMM and MC aren't direct price forecasts; we skip them for blending
    }
    regime = models.get("hmm", {}).get("current_regime", "unknown")
    blended = blend_forecasts(forecasts, last_price, regime=regime)

    # 6. Risk-adjusted exit recommendation
    latest_funding = float(funding["fundingRate"].iloc[0]) if len(funding) else None
    exit_rec = exit_signal(hist, regime=regime, funding_rate=latest_funding)

    # 7. Probability to hit 99k (from Monte Carlo)
    mc_probs = models.get("mc", {})

    # Print everything clearly
    print("===== BTC Quant Pipeline Summary =====")
    print(f"Last price: ${last_price:,.2f}")
    if rt_price:
        print(f"Realtime price (CoinGecko): ${rt_price:,.2f}")
    print("\n-- Risk Metrics --")
    for k, v in risk.items():
        print(f"{k.title():<15}: {v:.4f}" if isinstance(v, (int, float)) and np.isfinite(v) else f"{k.title():<15}: {v}")
    print("\n-- Regime --")
    print(f"Current regime: {regime}")
    probs = models.get("hmm", {}).get("probs")
    if probs is not None and isinstance(probs, np.ndarray) and probs.size:
        print(f"Regime probabilities: {np.round(probs, 3)} (order: bear/sideways/bull)")
    print("\n-- Model Forecasts (prices) --")
    print("ARIMA:", arima_fc)
    print("Prophet:", prophet_fc)
    print("LSTM next price:", lstm_next_price)
    print("GARCH vol forecast (percent):", models.get("garch", {}).get("vol_forecast"))
    print("\n-- Monte Carlo: Prob. hit $99k --")
    for h in (1, 7, 30):
        p = mc_probs.get(h)
        if p is not None:
            print(f"Within {h:>2}d: {p*100:.2f}%")
    print("\n-- Ensemble Blended Forecasts --")
    for k in ["1h", "1d", "1w", "1m"]:
        v = blended.get(k)
        print(f"{k:>2}: ${v:,.2f}" if v and np.isfinite(v) else f"{k:>2}: N/A")
    print("\n-- Exit Signal --")
    print(f"Signal: {'EXIT' if exit_rec['signal'] else 'HOLD'} (score={exit_rec['score']:.2f})")
    print("Reasons:", exit_rec["explanation"])


if __name__ == "__main__":
    main()
