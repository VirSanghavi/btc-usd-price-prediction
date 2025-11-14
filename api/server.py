from __future__ import annotations

import math
from typing import Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from crypto_quant.utils.config import set_seed
from crypto_quant.data.btc_history import load_btc_history
from crypto_quant.data.realtime import fetch_btc_price_coingecko
from crypto_quant.data.onchain import fetch_mvrv_z_score, fetch_nvt_ratio
from crypto_quant.data.derivatives import fetch_funding_rate, fetch_open_interest, fetch_long_short_ratio
from crypto_quant.data.sentiment import fetch_fear_greed_index
from crypto_quant.data.macro import fetch_dxy_yfinance, sp500_correlation_with_btc, fetch_fred_interest_rates
from crypto_quant.indicators.technical import (
    multi_timeframe_rsi,
    macd,
    stochastic_oscillator,
    bollinger_bands,
    atr,
    roc,
    vwap_deviation,
    fractal_dimension_index,
    hurst_exponent,
)
from crypto_quant.indicators.volatility import realized_volatility
from crypto_quant.models.arima_model import fit_arima_forecast
from crypto_quant.models.prophet_model import fit_prophet_forecast
from crypto_quant.models.lstm_model import lstm_predict_next_prices
from crypto_quant.models.hmm_regime import fit_hmm_regimes
from crypto_quant.models.monte_carlo import mc_hit_probability
from crypto_quant.models.garch_model import fit_garch_and_forecast
from crypto_quant.risk.risk_metrics import sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio
from crypto_quant.risk.exit_signals import exit_signal
from crypto_quant.ensemble.meta_learner import blend_forecasts


app = FastAPI(title="BTC Quant Forecast API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_close(df: pd.DataFrame) -> pd.DataFrame:
    if "close" not in df.columns and "price" in df.columns:
        df = df.copy()
        df["close"] = df["price"]
    return df


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = _ensure_close(df)
    out = d.sort_values("date").reset_index(drop=True)
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


@app.get("/api/forecast")
def forecast_endpoint() -> Dict[str, Any]:
    """Run the full pipeline and return JSON results suitable for UI consumption."""
    set_seed(42)
    resp: Dict[str, Any] = {"ok": True}
    try:
        # Data load + realtime
        hist = load_btc_history()
        hist = _ensure_close(hist)
        rt_price = fetch_btc_price_coingecko()
        if rt_price and not math.isnan(rt_price):
            last_date = hist["date"].max().normalize()
            today = pd.Timestamp.utcnow().normalize()
            if today > last_date:
                row = {"date": today, "close": float(rt_price), "price": float(rt_price)}
                hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)

        # Indicators + risk
        feats = _compute_indicators(hist)
        close = feats["close"].dropna()
        last_price = float(close.iloc[-1]) if len(close) else float("nan")
        rets = np.log(close).diff().dropna()
        risk = {
            "sharpe": sharpe_ratio(rets),
            "sortino": sortino_ratio(rets),
            "max_drawdown": max_drawdown(close),
            "calmar": calmar_ratio(rets, close),
        }

        # Side data (best-effort)
        funding = fetch_funding_rate().tail(1)

        # Models
        arima_fc = fit_arima_forecast(feats)
        prophet_fc = fit_prophet_forecast(feats)
        lstm_fc = lstm_predict_next_prices(feats)
        hmm = fit_hmm_regimes(feats)
        mc = mc_hit_probability(feats)
        garch = fit_garch_and_forecast(feats["ret"]) if "ret" in feats.columns else {"model": None, "vol_forecast": np.array([])}

        regime = hmm.get("current_regime", "unknown")
        latest_funding = float(funding["fundingRate"].iloc[0]) if len(funding) else None
        exit_rec = exit_signal(hist, regime=regime, funding_rate=latest_funding)

        forecasts = {
            "arima": arima_fc,
            "prophet": prophet_fc,
            "lstm": {1: float(lstm_fc.get("next_price", float("nan")))},
        }
        blended = blend_forecasts(forecasts, last_price, regime=regime)

        resp.update(
            last_price=last_price,
            realtime_price=rt_price,
            risk=risk,
            regime=regime,
            regime_probs=(hmm.get("probs").tolist() if isinstance(hmm.get("probs"), np.ndarray) else None),
            forecasts={
                "arima": arima_fc,
                "prophet": prophet_fc,
                "lstm_next_price": lstm_fc.get("next_price"),
                "garch_vol_forecast": (garch.get("vol_forecast").tolist() if isinstance(garch.get("vol_forecast"), np.ndarray) else None),
            },
            monte_carlo=mc,
            blended=blended,
            exit_signal=exit_rec,
        )
    except Exception as e:
        resp.update(ok=False, error=str(e))
    return resp
