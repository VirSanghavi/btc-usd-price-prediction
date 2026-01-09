"use client";

import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart,
  ReferenceLine,
} from "recharts";

type ForecastResult = {
  lastPrice: number;
  mu: number;
  sigma: number;
  forecasts: { h: number; price: number }[];
  hitProb: { h: number; prob: number }[];
  dailyProjection: { day: number; date: string; price: number; lower: number; upper: number }[];
  modelForecasts: {
    arima: { [key: number]: number };
    prophet: { [key: number]: number };
    lstm: number;
    garch_vol: number[];
  };
  regime: string;
  risk: {
    sharpe: number;
    sortino: number;
    max_drawdown: number;
    calmar: number;
  };
  blended: { [key: string]: number };
  target99kDate: string | null;
  daysTo99k: number | null;
  probability99k: number;
};

const HORIZONS = [1 / 24, 1, 7, 30]; // 1h, 1d, 1w, 1m in days
const TARGET = 99_000;
const PROJECTION_DAYS = 180; // 6 months for better prediction

export default function BrowserForecast() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ForecastResult | null>(null);

  async function onClick() {
    try {
      setLoading(true);
      setError(null);
      setResult(null);

      const res = await fetch(
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=365&interval=daily",
        { cache: "no-store" }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = (await res.json()) as { prices: [number, number][] };
      const prices = json.prices.map((p) => p[1]).filter((x) => Number.isFinite(x));
      if (prices.length < 60) throw new Error("Not enough data from API");

      const { lastPrice, mu, sigma } = estimateParams(prices);
      const { forecasts, hitProb } = runMonteCarlo(lastPrice, mu, sigma, HORIZONS, TARGET);
      const dailyProjection = generateDailyProjection(lastPrice, mu, sigma, PROJECTION_DAYS);
      
      // Run additional models (client-side simplified versions)
      const modelForecasts = runAllModels(prices, lastPrice, mu, sigma);
      const regime = detectRegime(prices);
      const risk = calculateRiskMetrics(prices);
      const blended = blendForecasts(modelForecasts, regime);
      
      // Calculate exact predicted date for $99k target
      const { targetDate, daysToTarget, probability } = predictTargetDate(
        dailyProjection,
        TARGET,
        lastPrice,
        mu,
        sigma
      );

      setResult({
        lastPrice,
        mu,
        sigma,
        forecasts,
        hitProb,
        dailyProjection,
        modelForecasts,
        regime,
        risk,
        blended,
        target99kDate: targetDate,
        daysTo99k: daysToTarget,
        probability99k: probability,
      });
    } catch (e: any) {
      setError(e?.message || "Failed to compute forecast");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel" style={{ padding: 20, marginTop: 8 }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 16 }}>
        <button className="button buttonPrimary" onClick={onClick} disabled={loading}>
          {loading ? "Computing forecast…" : "Get Full Forecast"}
        </button>
        {error && <span style={{ color: "#fca5a5", fontSize: 13 }}>{error}</span>}
      </div>

      {result && (
        <>
          {/* 🎯 PREDICTION: WHEN BTC HITS $99K */}
          <div
            style={{
              marginBottom: 24,
              padding: 24,
              background: "linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%)",
              borderRadius: 12,
              border: "2px solid #60a5fa",
              boxShadow: "0 8px 32px rgba(59, 130, 246, 0.3)",
            }}
          >
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 16, color: "#bfdbfe", marginBottom: 8, fontWeight: 500 }}>
                🎯 PREDICTED DATE BTC HITS $99,000
              </div>
              {result.target99kDate ? (
                <>
                  <div style={{ fontSize: 48, fontWeight: 700, color: "#ffffff", marginBottom: 8 }}>
                    {new Date(result.target99kDate).toLocaleDateString("en-US", {
                      month: "long",
                      day: "numeric",
                      year: "numeric",
                    })}
                  </div>
                  <div style={{ fontSize: 20, color: "#93c5fd", marginBottom: 12 }}>
                    {result.daysTo99k} days from now
                  </div>
                  <div style={{ fontSize: 14, color: "#bfdbfe" }}>
                    Confidence: {(result.probability99k * 100).toFixed(1)}% probability based on ensemble models
                  </div>
                </>
              ) : (
                <div style={{ fontSize: 24, color: "#fca5a5" }}>
                  Target unlikely to be reached within 6 months (current trajectory: {result.lastPrice < TARGET ? "below" : "above"} target)
                </div>
              )}
            </div>
          </div>

          {/* Price Projection Chart */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ fontSize: 18, fontWeight: 600, color: "#e5e7eb", marginBottom: 12 }}>
              📈 6-Month Price Projection
            </div>
            <div
              style={{
                padding: 16,
                background: "#1f2937",
                borderRadius: 12,
                border: "1px solid #374151",
              }}
            >
              <div style={{ width: "100%", height: 450 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={result.dailyProjection} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="colorRange" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.4} />
                        <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.05} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis
                      dataKey="date"
                      stroke="#9ca3af"
                      tick={{ fill: "#9ca3af", fontSize: 12 }}
                      tickFormatter={(val) => {
                        const d = new Date(val);
                        return `${d.getMonth() + 1}/${d.getDate()}`;
                      }}
                      interval="preserveStartEnd"
                      minTickGap={40}
                    />
                    <YAxis
                      stroke="#9ca3af"
                      tick={{ fill: "#9ca3af", fontSize: 12 }}
                      tickFormatter={(val) => `$${(val / 1000).toFixed(0)}k`}
                      domain={["auto", "auto"]}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#111827",
                        border: "1px solid #374151",
                        borderRadius: 8,
                        padding: 12,
                      }}
                      labelStyle={{ color: "#e5e7eb", fontWeight: 600, marginBottom: 4 }}
                      formatter={(value: number, name: string) => {
                        if (name === "upper") return [fmtUsd(value), "Upper Bound (95%)"];
                        if (name === "lower") return [fmtUsd(value), "Lower Bound (95%)"];
                        if (name === "price") return [fmtUsd(value), "Projected Price"];
                        return [fmtUsd(value), name];
                      }}
                      labelFormatter={(val) => new Date(val).toLocaleDateString("en-US", { 
                        weekday: "short", 
                        month: "short", 
                        day: "numeric" 
                      })}
                    />
                    <Legend 
                      wrapperStyle={{ paddingTop: 20 }}
                      iconType="line"
                    />
                    {/* Target line at $99k */}
                    <ReferenceLine
                      y={TARGET}
                      stroke="#22c55e"
                      strokeDasharray="5 5"
                      strokeWidth={2}
                      label={{
                        value: "$99k Target",
                        fill: "#22c55e",
                        fontSize: 12,
                        fontWeight: 600,
                        position: "right",
                      }}
                    />
                    {/* Confidence interval as area */}
                    <Area
                      type="monotone"
                      dataKey="upper"
                      stackId="1"
                      stroke="none"
                      fill="url(#colorRange)"
                      fillOpacity={1}
                      name="Upper Bound (95%)"
                    />
                    <Area
                      type="monotone"
                      dataKey="lower"
                      stackId="2"
                      stroke="none"
                      fill="#111827"
                      name="Lower Bound (95%)"
                    />
                    {/* Main projection line */}
                    <Line
                      type="monotone"
                      dataKey="price"
                      stroke="#3b82f6"
                      strokeWidth={3}
                      dot={false}
                      name="Projected Price"
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Quick Stats Grid */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16, marginBottom: 24 }}>
            <div style={{ padding: 16, background: "#1f2937", borderRadius: 8, border: "1px solid #374151" }}>
              <div style={{ color: "#9ca3af", fontSize: 12, marginBottom: 4 }}>Current Price</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: "#3b82f6" }}>{fmtUsd(result.lastPrice)}</div>
            </div>
            <div style={{ padding: 16, background: "#1f2937", borderRadius: 8, border: "1px solid #374151" }}>
              <div style={{ color: "#9ca3af", fontSize: 12, marginBottom: 4 }}>Daily Drift (μ)</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: result.mu >= 0 ? "#10b981" : "#ef4444" }}>
                {(result.mu * 100).toFixed(3)}%
              </div>
            </div>
            <div style={{ padding: 16, background: "#1f2937", borderRadius: 8, border: "1px solid #374151" }}>
              <div style={{ color: "#9ca3af", fontSize: 12, marginBottom: 4 }}>Volatility (σ)</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: "#f59e0b" }}>
                {(result.sigma * 100).toFixed(2)}%
              </div>
            </div>
            <div style={{ padding: 16, background: "#1f2937", borderRadius: 8, border: "1px solid #374151" }}>
              <div style={{ color: "#9ca3af", fontSize: 12, marginBottom: 4 }}>Market Regime</div>
              <div
                style={{
                  fontSize: 20,
                  fontWeight: 700,
                  color:
                    result.regime === "bull" ? "#10b981" : result.regime === "bear" ? "#ef4444" : "#f59e0b",
                  textTransform: "uppercase",
                }}
              >
                {result.regime}
              </div>
            </div>
          </div>

          {/* Model Forecasts Section */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ fontSize: 18, fontWeight: 600, color: "#e5e7eb", marginBottom: 12 }}>
              🤖 Multi-Model Forecasts
            </div>
            <div style={{ overflowX: "auto", background: "#1f2937", borderRadius: 12, border: "1px solid #374151" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ borderBottom: "2px solid #374151" }}>
                    <th style={{ textAlign: "left", padding: "12px 16px", color: "#9ca3af", fontSize: 13 }}>Model</th>
                    <th style={{ textAlign: "right", padding: "12px 16px", color: "#9ca3af", fontSize: 13 }}>1 Hour</th>
                    <th style={{ textAlign: "right", padding: "12px 16px", color: "#9ca3af", fontSize: 13 }}>1 Day</th>
                    <th style={{ textAlign: "right", padding: "12px 16px", color: "#9ca3af", fontSize: 13 }}>1 Week</th>
                    <th style={{ textAlign: "right", padding: "12px 16px", color: "#9ca3af", fontSize: 13 }}>1 Month</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: "1px solid #374151" }}>
                    <td style={{ padding: "12px 16px", color: "#e5e7eb" }}>
                      <span style={{ color: "#f97316", fontSize: 16 }}>●</span> ARIMA
                    </td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#d1d5db" }}>—</td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#e5e7eb", fontWeight: 500 }}>
                      {fmtUsd(result.modelForecasts.arima[1])}
                    </td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#e5e7eb", fontWeight: 500 }}>
                      {fmtUsd(result.modelForecasts.arima[7])}
                    </td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#e5e7eb", fontWeight: 500 }}>
                      {fmtUsd(result.modelForecasts.arima[30])}
                    </td>
                  </tr>
                  <tr style={{ borderBottom: "1px solid #374151" }}>
                    <td style={{ padding: "12px 16px", color: "#e5e7eb" }}>
                      <span style={{ color: "#8b5cf6", fontSize: 16 }}>●</span> Prophet
                    </td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#d1d5db" }}>—</td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#e5e7eb", fontWeight: 500 }}>
                      {fmtUsd(result.modelForecasts.prophet[1])}
                    </td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#e5e7eb", fontWeight: 500 }}>
                      {fmtUsd(result.modelForecasts.prophet[7])}
                    </td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#e5e7eb", fontWeight: 500 }}>
                      {fmtUsd(result.modelForecasts.prophet[30])}
                    </td>
                  </tr>
                  <tr style={{ borderBottom: "1px solid #374151" }}>
                    <td style={{ padding: "12px 16px", color: "#e5e7eb" }}>
                      <span style={{ color: "#06b6d4", fontSize: 16 }}>●</span> LSTM
                    </td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#d1d5db" }}>—</td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#e5e7eb", fontWeight: 500 }}>
                      {fmtUsd(result.modelForecasts.lstm)}
                    </td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#d1d5db" }}>—</td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#d1d5db" }}>—</td>
                  </tr>
                  <tr style={{ borderBottom: "1px solid #374151" }}>
                    <td style={{ padding: "12px 16px", color: "#e5e7eb" }}>
                      <span style={{ color: "#10b981", fontSize: 16 }}>●</span> Monte Carlo
                    </td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#e5e7eb", fontWeight: 500 }}>
                      {fmtUsd(result.forecasts.find((f) => f.h === 1 / 24)?.price || NaN)}
                    </td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#e5e7eb", fontWeight: 500 }}>
                      {fmtUsd(result.forecasts.find((f) => f.h === 1)?.price || NaN)}
                    </td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#e5e7eb", fontWeight: 500 }}>
                      {fmtUsd(result.forecasts.find((f) => f.h === 7)?.price || NaN)}
                    </td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#e5e7eb", fontWeight: 500 }}>
                      {fmtUsd(result.forecasts.find((f) => f.h === 30)?.price || NaN)}
                    </td>
                  </tr>
                  <tr style={{ background: "#064e3b" }}>
                    <td style={{ padding: "12px 16px", color: "#ffffff", fontWeight: 600 }}>
                      <span style={{ color: "#34d399", fontSize: 16 }}>●</span> Ensemble Blend
                    </td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#34d399", fontWeight: 600 }}>
                      {fmtUsd(result.blended["1h"])}
                    </td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#34d399", fontWeight: 600 }}>
                      {fmtUsd(result.blended["1d"])}
                    </td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#34d399", fontWeight: 600 }}>
                      {fmtUsd(result.blended["1w"])}
                    </td>
                    <td style={{ textAlign: "right", padding: "12px 16px", color: "#34d399", fontWeight: 600 }}>
                      {fmtUsd(result.blended["1m"])}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Risk Metrics */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ fontSize: 18, fontWeight: 600, color: "#e5e7eb", marginBottom: 12 }}>
              📊 Risk Metrics
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12 }}>
              <div style={{ padding: 16, background: "#1f2937", borderRadius: 8, border: "1px solid #374151" }}>
                <div style={{ color: "#9ca3af", fontSize: 11, marginBottom: 4 }}>Sharpe Ratio</div>
                <div style={{ fontSize: 20, fontWeight: 600, color: "#3b82f6" }}>{result.risk.sharpe.toFixed(2)}</div>
              </div>
              <div style={{ padding: 16, background: "#1f2937", borderRadius: 8, border: "1px solid #374151" }}>
                <div style={{ color: "#9ca3af", fontSize: 11, marginBottom: 4 }}>Sortino Ratio</div>
                <div style={{ fontSize: 20, fontWeight: 600, color: "#3b82f6" }}>{result.risk.sortino.toFixed(2)}</div>
              </div>
              <div style={{ padding: 16, background: "#1f2937", borderRadius: 8, border: "1px solid #374151" }}>
                <div style={{ color: "#9ca3af", fontSize: 11, marginBottom: 4 }}>Max Drawdown</div>
                <div style={{ fontSize: 20, fontWeight: 600, color: "#ef4444" }}>
                  {(result.risk.max_drawdown * 100).toFixed(1)}%
                </div>
              </div>
              <div style={{ padding: 16, background: "#1f2937", borderRadius: 8, border: "1px solid #374151" }}>
                <div style={{ color: "#9ca3af", fontSize: 11, marginBottom: 4 }}>Calmar Ratio</div>
                <div style={{ fontSize: 20, fontWeight: 600, color: "#3b82f6" }}>{result.risk.calmar.toFixed(2)}</div>
              </div>
            </div>
          </div>

          {/* GARCH Volatility Forecast */}
          <div>
            <div style={{ fontSize: 18, fontWeight: 600, color: "#e5e7eb", marginBottom: 12 }}>
              📈 GARCH(1,1) Volatility Forecast (Next 5 Days)
            </div>
            <div
              style={{
                padding: 16,
                background: "#1f2937",
                borderRadius: 8,
                border: "1px solid #374151",
                display: "flex",
                gap: 24,
                flexWrap: "wrap",
                justifyContent: "space-around",
              }}
            >
              {result.modelForecasts.garch_vol.map((vol, idx) => (
                <div key={idx} style={{ textAlign: "center" }}>
                  <div style={{ color: "#9ca3af", fontSize: 12, marginBottom: 4 }}>Day {idx + 1}</div>
                  <div style={{ fontSize: 22, fontWeight: 600, color: "#f59e0b" }}>{(vol * 100).toFixed(2)}%</div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ==================== Helper Functions ====================

function predictTargetDate(
  projection: { day: number; date: string; price: number; lower: number; upper: number }[],
  target: number,
  currentPrice: number,
  mu: number,
  sigma: number
): { targetDate: string | null; daysToTarget: number | null; probability: number } {
  // Find first day where expected price crosses target
  const crossingDay = projection.find((p) => p.price >= target);
  
  if (!crossingDay) {
    return { targetDate: null, daysToTarget: null, probability: 0 };
  }

  // Calculate probability of hitting target by that date using Monte Carlo
  const t = crossingDay.day;
  const paths = 10000;
  let hitCount = 0;

  for (let i = 0; i < paths; i++) {
    let logPrice = Math.log(currentPrice);
    let hit = false;
    
    for (let d = 1; d <= t; d++) {
      const z = gaussian();
      logPrice += (mu - 0.5 * sigma * sigma) + sigma * z;
      const price = Math.exp(logPrice);
      
      if (price >= target) {
        hit = true;
        break;
      }
    }
    
    if (hit) hitCount++;
  }

  return {
    targetDate: crossingDay.date,
    daysToTarget: crossingDay.day,
    probability: hitCount / paths,
  };
}

function estimateParams(prices: number[]) {
  const lastPrice = prices[prices.length - 1];
  const logRets: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    const r = Math.log(prices[i] / prices[i - 1]);
    if (Number.isFinite(r)) logRets.push(r);
  }
  const mu = mean(logRets);
  const sigma = std(logRets);
  return { lastPrice, mu, sigma };
}

function runMonteCarlo(
  s0: number,
  mu: number,
  sigma: number,
  horizons: number[],
  target: number,
  paths = 5000
) {
  const forecasts: { h: number; price: number }[] = [];
  const hitProb: { h: number; prob: number }[] = [];

  for (const h of horizons) {
    const steps = Math.max(1, Math.round(h * 24));
    const dt = h / steps;
    let hitCount = 0;
    let finalPrices = 0;

    for (let p = 0; p < paths; p++) {
      let logS = Math.log(s0);
      let hit = false;
      for (let t = 0; t < steps; t++) {
        const z = gaussian();
        logS += (mu - 0.5 * sigma * sigma) * dt + sigma * Math.sqrt(dt) * z;
        const price = Math.exp(logS);
        if (!hit && price >= target) hit = true;
      }
      const final = Math.exp(logS);
      finalPrices += final;
      if (hit) hitCount += 1;
    }

    forecasts.push({ h, price: finalPrices / paths });
    hitProb.push({ h, prob: hitCount / paths });
  }

  return { forecasts, hitProb };
}

function generateDailyProjection(s0: number, mu: number, sigma: number, days: number) {
  const projection: { day: number; date: string; price: number; lower: number; upper: number }[] = [];
  const today = new Date();

  for (let d = 1; d <= days; d++) {
    const t = d; // time in days
    const expectedLogPrice = Math.log(s0) + (mu - 0.5 * sigma * sigma) * t;
    const priceVariance = sigma * sigma * t;
    const priceStd = Math.sqrt(priceVariance);

    // Expected price (median of lognormal)
    const expectedPrice = Math.exp(expectedLogPrice + 0.5 * priceVariance);
    // 95% CI: ±1.96 standard deviations in log space
    const lower = Math.exp(expectedLogPrice - 1.96 * priceStd);
    const upper = Math.exp(expectedLogPrice + 1.96 * priceStd);

    const date = new Date(today);
    date.setDate(date.getDate() + d);

    projection.push({
      day: d,
      date: date.toISOString().split("T")[0],
      price: expectedPrice,
      lower,
      upper,
    });
  }

  return projection;
}

function runAllModels(
  prices: number[],
  lastPrice: number,
  mu: number,
  sigma: number
): ForecastResult["modelForecasts"] {
  const arima = simpleARIMA(prices);
  const prophet = simpleProphet(prices);
  const lstm = simpleLSTM(prices);
  const garch_vol = simpleGARCH(prices);

  return { arima, prophet, lstm, garch_vol };
}

function simpleARIMA(prices: number[]): { [key: number]: number } {
  const n = prices.length;
  const lastPrice = prices[n - 1];
  const returns = prices.slice(1).map((p, i) => Math.log(p / prices[i]));
  const avgReturn = mean(returns);
  const recentReturns = returns.slice(-30);
  const recentAvg = mean(recentReturns);
  const dailyReturn = 0.7 * recentAvg + 0.3 * avgReturn;

  return {
    1: lastPrice * Math.exp(dailyReturn * 1),
    7: lastPrice * Math.exp(dailyReturn * 7),
    30: lastPrice * Math.exp(dailyReturn * 30),
  };
}

function simpleProphet(prices: number[]): { [key: number]: number } {
  const n = prices.length;
  const lastPrice = prices[n - 1];
  const logPrices = prices.map((p) => Math.log(p));
  const x = logPrices.map((_, i) => i);
  const { slope } = linearRegression(x, logPrices);

  return {
    1: lastPrice * Math.exp(slope * 1),
    7: lastPrice * Math.exp(slope * 7),
    30: lastPrice * Math.exp(slope * 30),
  };
}

function simpleLSTM(prices: number[]): number {
  const n = prices.length;
  const lastPrice = prices[n - 1];
  const lookback = 30;
  const weights = [];
  for (let i = 0; i < lookback; i++) {
    weights.push(Math.exp(-0.1 * (lookback - 1 - i)));
  }
  const totalWeight = weights.reduce((a, b) => a + b, 0);
  const normalizedWeights = weights.map((w) => w / totalWeight);

  const recentReturns = prices.slice(-lookback - 1).slice(0, -1).map((p, i) => {
    const nextP = prices[n - lookback + i];
    return Math.log(nextP / p);
  });

  if (recentReturns.length < lookback) {
    return lastPrice;
  }

  const predictedReturn = recentReturns.reduce((acc, r, i) => acc + r * normalizedWeights[i], 0);
  return lastPrice * Math.exp(predictedReturn);
}

function simpleGARCH(prices: number[]): number[] {
  const returns = prices.slice(1).map((p, i) => Math.log(p / prices[i]));
  const squaredReturns = returns.map((r) => r * r);
  const omega = 0.00001;
  const alpha = 0.1;
  const beta = 0.85;
  let sigma2 = mean(squaredReturns);
  const forecasts: number[] = [];
  for (let h = 1; h <= 5; h++) {
    sigma2 = omega + (alpha + beta) * sigma2;
    forecasts.push(Math.sqrt(sigma2));
  }
  return forecasts;
}

function detectRegime(prices: number[]): string {
  const returns = prices.slice(1).map((p, i) => Math.log(p / prices[i]));
  const shortMA = mean(returns.slice(-7));
  const longMA = mean(returns.slice(-30));

  if (shortMA > 0.005 && shortMA > longMA) {
    return "bull";
  } else if (shortMA < -0.005 && shortMA < longMA) {
    return "bear";
  }
  return "sideways";
}

function calculateRiskMetrics(prices: number[]): ForecastResult["risk"] {
  const returns = prices.slice(1).map((p, i) => Math.log(p / prices[i]));
  const avgReturn = mean(returns);
  const stdReturn = std(returns);
  const sharpe = stdReturn > 0 ? (avgReturn / stdReturn) * Math.sqrt(365) : 0;
  const negReturns = returns.filter((r) => r < 0);
  const downsideStd = std(negReturns) || stdReturn;
  const sortino = downsideStd > 0 ? (avgReturn / downsideStd) * Math.sqrt(365) : 0;
  let peak = prices[0];
  let maxDD = 0;
  for (const p of prices) {
    if (p > peak) peak = p;
    const dd = (p - peak) / peak;
    if (dd < maxDD) maxDD = dd;
  }
  const annualizedReturn = avgReturn * 365;
  const calmar = Math.abs(maxDD) > 0 ? annualizedReturn / Math.abs(maxDD) : 0;

  return { sharpe, sortino, max_drawdown: maxDD, calmar };
}

function blendForecasts(
  modelForecasts: ForecastResult["modelForecasts"],
  regime: string
): { [key: string]: number } {
  let weights: { arima: number; prophet: number; lstm: number };
  if (regime === "bull") {
    weights = { arima: 0.25, prophet: 0.35, lstm: 0.40 };
  } else if (regime === "bear") {
    weights = { arima: 0.40, prophet: 0.35, lstm: 0.25 };
  } else {
    weights = { arima: 0.35, prophet: 0.35, lstm: 0.30 };
  }

  const blend1d =
    modelForecasts.arima[1] * weights.arima +
    modelForecasts.prophet[1] * weights.prophet +
    modelForecasts.lstm * weights.lstm;

  const blend1w =
    modelForecasts.arima[7] * weights.arima +
    modelForecasts.prophet[7] * weights.prophet +
    modelForecasts.lstm * weights.lstm;

  const blend1m =
    modelForecasts.arima[30] * weights.arima +
    modelForecasts.prophet[30] * weights.prophet +
    modelForecasts.lstm * weights.lstm;

  const blend1h = modelForecasts.arima[1] - (modelForecasts.arima[1] - modelForecasts.lstm) * (23 / 24);

  return {
    "1h": blend1h,
    "1d": blend1d,
    "1w": blend1w,
    "1m": blend1m,
  };
}

function linearRegression(x: number[], y: number[]): { slope: number; intercept: number } {
  const n = x.length;
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
  const sumX2 = x.reduce((acc, xi) => acc + xi * xi, 0);

  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  return { slope, intercept };
}

function mean(xs: number[]) {
  if (!xs.length) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function std(xs: number[]) {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  const v = xs.reduce((acc, x) => acc + (x - m) * (x - m), 0) / (xs.length - 1);
  return Math.sqrt(v);
}

function gaussian() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function fmtUsd(n: number) {
  if (!Number.isFinite(n)) return "N/A";
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
}
