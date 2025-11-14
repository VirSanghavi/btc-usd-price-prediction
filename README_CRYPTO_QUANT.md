# crypto_quant: Bitcoin Quantitative Forecasting System

Professional-grade, multi-module quant research pipeline inspired by hedge fund architectures.

## Structure

```
crypto_quant/
  main.py
  utils/
    __init__.py
    config.py
  data/
    __init__.py
    btc_history.py
    realtime.py
    onchain.py
    derivatives.py
    sentiment.py
    macro.py
  indicators/
    __init__.py
    technical.py
    volatility.py
    microstructure.py
  models/
    __init__.py
    arima_model.py
    prophet_model.py
    lstm_model.py
    hmm_regime.py
    monte_carlo.py
    garch_model.py
  ensemble/
    __init__.py
    meta_learner.py
  risk/
    __init__.py
    risk_metrics.py
    exit_signals.py
```

## Setup

Install dependencies (Python 3.10+ recommended):

```bash
pip install pandas numpy yfinance scikit-learn pmdarima prophet torch hmmlearn arch kagglehub
```

Optional environment variables for APIs:

- `GLASSNODE_API_KEY`: enables on-chain metrics (MVRV Z-score, NVT)
- `FRED_API_KEY`: enables US interest rates from FRED (DGS2, DGS10)
- `BTC_HIST_CSV`: path to a CSV file with daily BTC data

## Run

```bash
python -m crypto_quant.main
```

This will:

- Load historical BTC (Kaggle/CSV/yfinance fallback)
- Fetch realtime price (CoinGecko)
- Compute indicators and risk metrics
- Run models: ARIMA, Prophet, LSTM, HMM regimes, Monte Carlo, GARCH
- Blend forecasts via meta-learner (1h/1d/1w/1m)
- Print regime, hit-99k probabilities, and an exit recommendation

## Notes

- Network calls are best-effort with graceful fallbacks (empty frames when keys are missing).
- LSTM is intentionally small and trains briefly to keep runtime reasonable.
- If some libraries are not installed, corresponding outputs will be `NaN` or skipped gracefully.
