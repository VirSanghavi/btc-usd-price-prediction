# ---------------------------------------
# Bitcoin Trading Signal + Prediction Tool
# ---------------------------------------
# pip install kagglehub[pandas-datasets] pandas numpy requests scikit-learn

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ------------------------------
# 1. LOAD HISTORICAL BTC DATA
# ------------------------------

file_path = "bitcoin_data.csv"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "mczielinski/bitcoin-historical-data",
    file_path,
)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp")
df["Price"] = df["Close"]

# ------------------------------
# 2. GET CURRENT BTC PRICE
# ------------------------------
def get_current_btc_price():
    try:
        r = requests.get("https://api.coindesk.com/v1/bpi/currentprice.json")
        return r.json()["bpi"]["USD"]["rate_float"]
    except:
        return None

current_price = get_current_btc_price()
today = datetime.now().date()

print("\n===== Current Bitcoin Price =====")
print(f"Date: {today}")
print(f"Current Price (USD): {current_price}")

# ------------------------------
# 3. TECHNICAL INDICATORS
# ------------------------------

# Moving averages
df["MA20"] = df["Price"].rolling(20).mean()
df["MA50"] = df["Price"].rolling(50).mean()
df["MA100"] = df["Price"].rolling(100).mean()
df["MA200"] = df["Price"].rolling(200).mean()

# RSI
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df["RSI"] = compute_RSI(df["Price"])

# MACD
df["EMA12"] = df["Price"].ewm(span=12, adjust=False).mean()
df["EMA26"] = df["Price"].ewm(span=26, adjust=False).mean()
df["MACD"] = df["EMA12"] - df["EMA26"]
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

# Volatility
df["DailyReturn"] = df["Price"].pct_change()
volatility = df["DailyReturn"].std() * np.sqrt(365)

print("\n===== Technical Indicators =====")
latest = df.iloc[-1]
print(f"20-day MA:  {latest['MA20']:.2f}")
print(f"50-day MA:  {latest['MA50']:.2f}")
print(f"100-day MA: {latest['MA100']:.2f}")
print(f"200-day MA: {latest['MA200']:.2f}")
print(f"RSI:        {latest['RSI']:.2f}")
print(f"MACD:       {latest['MACD']:.2f}")
print(f"Signal:     {latest['Signal']:.2f}")
print(f"Volatility: {volatility:.4f}")

# ------------------------------
# 4. TRADING SIGNAL (PULL-OUT LOGIC)
# ------------------------------

def trading_recommendation(row):
    rsi = row["RSI"]
    price = row["Price"]
    ma50 = row["MA50"]
    ma200 = row["MA200"]
    macd = row["MACD"]
    signal = row["Signal"]

    signals = []

    # Overbought
    if rsi > 70:
        signals.append("RSI over 70: Market overbought")

    # Downtrend confirmation
    if price < ma50 < ma200:
        signals.append("Price below MA50 and MA200: Strong downtrend")

    # MACD bearish crossover
    if macd < signal:
        signals.append("MACD below signal: bearish momentum")

    if len(signals) == 0:
        return "No major pull-out signals right now"
    else:
        return "Potential exit signals:\n" + "\n".join(signals)

rec = trading_recommendation(latest)

print("\n===== Practical Trading Recommendation =====")
print(rec)

# ------------------------------
# 5. FORECAST TIME TO REACH TARGET PRICE
# ------------------------------

TARGET = 99000  # USD
df["LogPrice"] = np.log(df["Price"])
df["Days"] = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.days

valid = df.dropna(subset=["LogPrice", "Days"])
X = valid["Days"].values.reshape(-1, 1)
y = valid["LogPrice"].values

model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_
log_target = np.log(TARGET)

if slope <= 0:
    print("\nModel cannot predict return to 99K (trend is flat/downward).")
else:
    future_days = int((log_target - intercept) / slope)
    predicted_date = df["Timestamp"].iloc[0] + timedelta(days=future_days)

    print("\n===== Forecast: Days Until Bitcoin Reaches 99K =====")
    print(f"Estimated days: {future_days}")
    print(f"Predicted date: {predicted_date.date()}")

# ------------------------------
# 6. SUMMARY FOR YOUR TEACHER
# ------------------------------

print("\n===== Summary for Decision Making =====")

print(f"""
1. Current BTC Price: {current_price}

2. Trend Indicators:
   - BTC is {'above' if latest['Price'] > latest['MA200'] else 'below'} the 200-day average.
   - RSI is {latest['RSI']:.2f} (Overbought > 70, Oversold < 30)
   - MACD trend: {'Bullish' if latest['MACD'] > latest['Signal'] else 'Bearish'}

3. Volatility:
   - Annualized volatility: {volatility:.2f}

4. Recommendation:
   {rec}

5. Expected return to 99K:
   {predicted_date.date() if slope > 0 else 'Trend does not support recovery prediction'}
""")
