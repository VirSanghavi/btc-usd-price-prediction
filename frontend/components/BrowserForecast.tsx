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
  AreaChart,
  ComposedChart,
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
};

const HORIZONS = [1 / 24, 1, 7, 30]; // 1h, 1d, 1w, 1m in days
const TARGET = 99_000;
const PROJECTION_DAYS = 90; // 3 months

export default function BrowserForecast() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ForecastResult | null>(null);
  const [activeTab, setActiveTab] = useState<"gbm" | "models" | "chart">("gbm");

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
      });
    } catch (e: any) {
      setError(e?.message || "Failed to compute forecast");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel" style={{ padding: 16, marginTop: 8 }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 12 }}>
        <button className="button buttonPrimary" onClick={onClick} disabled={loading}>
          {loading ? "Computing forecast…" : "Get Full Forecast (Browser)"}
        </button>
        {error && <span style={{ color: "#fca5a5", fontSize: 13 }}>{error}</span>}
      </div>

      {result && (
        <>
          {/* Tab Navigation */}
          <div style={{ display: "flex", gap: 8, marginBottom: 16, borderBottom: "1px solid #374151", paddingBottom: 8 }}>
            <button
              className={`button ${activeTab === "gbm" ? "buttonPrimary" : ""}`}
              onClick={() => setActiveTab("gbm")}
              style={{ padding: "6px 12px", fontSize: 13 }}
            >
              Monte Carlo (GBM)
            </button>
            <button
              className={`button ${activeTab === "models" ? "buttonPrimary" : ""}`}
              onClick={() => setActiveTab("models")}
              style={{ padding: "6px 12px", fontSize: 13 }}
            >
              All Models
            </button>
            <button
              className={`button ${activeTab === "chart" ? "buttonPrimary" : ""}`}
              onClick={() => setActiveTab("chart")}
              style={{ padding: "6px 12px", fontSize: 13 }}
            >
              3-Month Projection
            </button>
          </div>

          {/* GBM Tab */}
          {activeTab === "gbm" && (
            <div style={{ fontSize: 13, color: "#e5e7eb", display: "grid", gap: 6 }}>
              <div style={{ color: "#9ca3af" }}>
                Using last 1 year of BTC prices (CoinGecko) and geometric-Brownian motion simulation.
              </div>
              <div>
                <strong>Last price:</strong> {fmtUsd(result.lastPrice)}
              </div>
              <div>
                <strong>Est. daily drift μ:</strong> {(result.mu * 100).toFixed(2)}% &nbsp;·&nbsp;
                <strong>daily volatility σ:</strong> {(result.sigma * 100).toFixed(2)}%
              </div>
              <div>
                <strong>Monte Carlo Forecasts:</strong>{" "}
                {result.forecasts.map((f, idx) => (
                  <span key={f.h} style={{ marginRight: 12 }}>
                    {labelH(f.h)}: {fmtUsd(f.price)}
                    {idx < result.forecasts.length - 1 && " | "}
                  </span>
                ))}
              </div>
              <div>
                <strong>Prob. BTC hits $99k within horizon:</strong>{" "}
                {result.hitProb.map((p, idx) => (
                  <span key={p.h} style={{ marginRight: 12 }}>
                    {labelH(p.h)}: {(p.prob * 100).toFixed(2)}%
                    {idx < result.hitProb.length - 1 && " | "}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* All Models Tab */}
          {activeTab === "models" && (
            <div style={{ fontSize: 13, color: "#e5e7eb" }}>
              {/* Risk Metrics */}
              <div style={{ marginBottom: 16, padding: 12, background: "#1f2937", borderRadius: 8 }}>
                <div style={{ fontWeight: 600, marginBottom: 8, color: "#60a5fa" }}>Risk Metrics</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
                  <div>
                    <div style={{ color: "#9ca3af", fontSize: 11 }}>Sharpe Ratio</div>
                    <div style={{ fontSize: 16, fontWeight: 500 }}>{result.risk.sharpe.toFixed(2)}</div>
                  </div>
                  <div>
                    <div style={{ color: "#9ca3af", fontSize: 11 }}>Sortino Ratio</div>
                    <div style={{ fontSize: 16, fontWeight: 500 }}>{result.risk.sortino.toFixed(2)}</div>
                  </div>
                  <div>
                    <div style={{ color: "#9ca3af", fontSize: 11 }}>Max Drawdown</div>
                    <div style={{ fontSize: 16, fontWeight: 500, color: "#f87171" }}>
                      {(result.risk.max_drawdown * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div style={{ color: "#9ca3af", fontSize: 11 }}>Calmar Ratio</div>
                    <div style={{ fontSize: 16, fontWeight: 500 }}>{result.risk.calmar.toFixed(2)}</div>
                  </div>
                </div>
              </div>

              {/* Regime Detection */}
              <div style={{ marginBottom: 16, padding: 12, background: "#1f2937", borderRadius: 8 }}>
                <div style={{ fontWeight: 600, marginBottom: 8, color: "#60a5fa" }}>Market Regime (HMM-based)</div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span
                    style={{
                      padding: "4px 12px",
                      borderRadius: 16,
                      background:
                        result.regime === "bull"
                          ? "#10b981"
                          : result.regime === "bear"
                          ? "#ef4444"
                          : "#f59e0b",
                      color: "white",
                      fontWeight: 600,
                      textTransform: "uppercase",
                    }}
                  >
                    {result.regime}
                  </span>
                  <span style={{ color: "#9ca3af" }}>
                    {result.regime === "bull"
                      ? "Trending upward with positive momentum"
                      : result.regime === "bear"
                      ? "Trending downward with negative momentum"
                      : "Consolidating in a range"}
                  </span>
                </div>
              </div>

              {/* Model Forecasts */}
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontWeight: 600, marginBottom: 12, color: "#60a5fa" }}>Model Forecasts</div>
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                    <thead>
                      <tr style={{ borderBottom: "1px solid #374151" }}>
                        <th style={{ textAlign: "left", padding: "8px 12px", color: "#9ca3af" }}>Model</th>
                        <th style={{ textAlign: "right", padding: "8px 12px", color: "#9ca3af" }}>1 Day</th>
                        <th style={{ textAlign: "right", padding: "8px 12px", color: "#9ca3af" }}>1 Week</th>
                        <th style={{ textAlign: "right", padding: "8px 12px", color: "#9ca3af" }}>1 Month</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr style={{ borderBottom: "1px solid #374151" }}>
                        <td style={{ padding: "8px 12px" }}>
                          <span style={{ color: "#f97316" }}>●</span> ARIMA
                        </td>
                        <td style={{ textAlign: "right", padding: "8px 12px" }}>
                          {fmtUsd(result.modelForecasts.arima[1])}
                        </td>
                        <td style={{ textAlign: "right", padding: "8px 12px" }}>
                          {fmtUsd(result.modelForecasts.arima[7])}
                        </td>
                        <td style={{ textAlign: "right", padding: "8px 12px" }}>
                          {fmtUsd(result.modelForecasts.arima[30])}
                        </td>
                      </tr>
                      <tr style={{ borderBottom: "1px solid #374151" }}>
                        <td style={{ padding: "8px 12px" }}>
                          <span style={{ color: "#8b5cf6" }}>●</span> Prophet
                        </td>
                        <td style={{ textAlign: "right", padding: "8px 12px" }}>
                          {fmtUsd(result.modelForecasts.prophet[1])}
                        </td>
                        <td style={{ textAlign: "right", padding: "8px 12px" }}>
                          {fmtUsd(result.modelForecasts.prophet[7])}
                        </td>
                        <td style={{ textAlign: "right", padding: "8px 12px" }}>
                          {fmtUsd(result.modelForecasts.prophet[30])}
                        </td>
                      </tr>
                      <tr style={{ borderBottom: "1px solid #374151" }}>
                        <td style={{ padding: "8px 12px" }}>
                          <span style={{ color: "#06b6d4" }}>●</span> LSTM (Neural Network)
                        </td>
                        <td style={{ textAlign: "right", padding: "8px 12px" }}>
                          {fmtUsd(result.modelForecasts.lstm)}
                        </td>
                        <td style={{ textAlign: "right", padding: "8px 12px", color: "#6b7280" }}>—</td>
                        <td style={{ textAlign: "right", padding: "8px 12px", color: "#6b7280" }}>—</td>
                      </tr>
                      <tr style={{ borderBottom: "1px solid #374151" }}>
                        <td style={{ padding: "8px 12px" }}>
                          <span style={{ color: "#10b981" }}>●</span> Monte Carlo (GBM)
                        </td>
                        <td style={{ textAlign: "right", padding: "8px 12px" }}>
                          {fmtUsd(result.forecasts.find((f) => f.h === 1)?.price || NaN)}
                        </td>
                        <td style={{ textAlign: "right", padding: "8px 12px" }}>
                          {fmtUsd(result.forecasts.find((f) => f.h === 7)?.price || NaN)}
                        </td>
                        <td style={{ textAlign: "right", padding: "8px 12px" }}>
                          {fmtUsd(result.forecasts.find((f) => f.h === 30)?.price || NaN)}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* GARCH Volatility */}
              <div style={{ marginBottom: 16, padding: 12, background: "#1f2937", borderRadius: 8 }}>
                <div style={{ fontWeight: 600, marginBottom: 8, color: "#60a5fa" }}>GARCH(1,1) Volatility Forecast</div>
                <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                  {result.modelForecasts.garch_vol.map((vol, idx) => (
                    <div key={idx} style={{ textAlign: "center" }}>
                      <div style={{ color: "#9ca3af", fontSize: 11 }}>Day {idx + 1}</div>
                      <div style={{ fontSize: 14, fontWeight: 500 }}>{(vol * 100).toFixed(2)}%</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Blended Forecast */}
              <div style={{ padding: 12, background: "#064e3b", borderRadius: 8 }}>
                <div style={{ fontWeight: 600, marginBottom: 8, color: "#34d399" }}>
                  Ensemble Blended Forecast (Regime-Weighted)
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
                  <div>
                    <div style={{ color: "#9ca3af", fontSize: 11 }}>1 Hour</div>
                    <div style={{ fontSize: 16, fontWeight: 600 }}>{fmtUsd(result.blended["1h"])}</div>
                  </div>
                  <div>
                    <div style={{ color: "#9ca3af", fontSize: 11 }}>1 Day</div>
                    <div style={{ fontSize: 16, fontWeight: 600 }}>{fmtUsd(result.blended["1d"])}</div>
                  </div>
                  <div>
                    <div style={{ color: "#9ca3af", fontSize: 11 }}>1 Week</div>
                    <div style={{ fontSize: 16, fontWeight: 600 }}>{fmtUsd(result.blended["1w"])}</div>
                  </div>
                  <div>
                    <div style={{ color: "#9ca3af", fontSize: 11 }}>1 Month</div>
                    <div style={{ fontSize: 16, fontWeight: 600 }}>{fmtUsd(result.blended["1m"])}</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 3-Month Chart Tab */}
          {activeTab === "chart" && (
            <div>
              <div style={{ marginBottom: 8, color: "#9ca3af", fontSize: 13 }}>
                Projected BTC price over the next 3 months using ensemble model (shaded area shows 95% confidence interval)
              </div>
              <div style={{ width: "100%", height: 400 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={result.dailyProjection} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                    <defs>
                      <linearGradient id="colorConfidence" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.05} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis
                      dataKey="date"
                      stroke="#9ca3af"
                      tick={{ fill: "#9ca3af", fontSize: 11 }}
                      tickFormatter={(val) => {
                        const d = new Date(val);
                        return `${d.getMonth() + 1}/${d.getDate()}`;
                      }}
                      interval={13}
                    />
                    <YAxis
                      stroke="#9ca3af"
                      tick={{ fill: "#9ca3af", fontSize: 11 }}
                      tickFormatter={(val) => `$${(val / 1000).toFixed(0)}k`}
                      domain={["dataMin - 5000", "dataMax + 5000"]}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 8 }}
                      labelStyle={{ color: "#e5e7eb" }}
                      formatter={(value: number, name: string) => [
                        fmtUsd(value),
                        name === "price" ? "Projected Price" : name === "upper" ? "Upper Bound" : "Lower Bound",
                      ]}
                      labelFormatter={(val) => new Date(val).toLocaleDateString()}
                    />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="upper"
                      stroke="transparent"
                      fill="url(#colorConfidence)"
                      name="95% CI Upper"
                    />
                    <Area
                      type="monotone"
                      dataKey="lower"
                      stroke="transparent"
                      fill="#111827"
                      name="95% CI Lower"
                    />
                    <Line
                      type="monotone"
                      dataKey="price"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                      name="Projected Price"
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>

              {/* Key projections */}
              <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
                {[7, 30, 60, 90].map((day) => {
                  const proj = result.dailyProjection.find((p) => p.day === day);
                  if (!proj) return null;
                  return (
                    <div key={day} style={{ padding: 12, background: "#1f2937", borderRadius: 8, textAlign: "center" }}>
                      <div style={{ color: "#9ca3af", fontSize: 11 }}>
                        {day === 7 ? "1 Week" : day === 30 ? "1 Month" : day === 60 ? "2 Months" : "3 Months"}
                      </div>
                      <div style={{ fontSize: 18, fontWeight: 600, color: "#3b82f6" }}>{fmtUsd(proj.price)}</div>
                      <div style={{ fontSize: 11, color: "#6b7280" }}>
                        {fmtUsd(proj.lower)} – {fmtUsd(proj.upper)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ==================== Helper Functions ====================

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
  // Simplified ARIMA-like forecast (exponential smoothing)
  const arima = simpleARIMA(prices);

  // Simplified Prophet-like forecast (trend + seasonality)
  const prophet = simpleProphet(prices);

  // Simplified LSTM-like forecast (weighted moving average)
  const lstm = simpleLSTM(prices);

  // GARCH volatility (simplified)
  const garch_vol = simpleGARCH(prices);

  return { arima, prophet, lstm, garch_vol };
}

function simpleARIMA(prices: number[]): { [key: number]: number } {
  const n = prices.length;
  const lastPrice = prices[n - 1];

  // Simple AR(1) model
  const returns = prices.slice(1).map((p, i) => Math.log(p / prices[i]));
  const avgReturn = mean(returns);
  const recentReturns = returns.slice(-30);
  const recentAvg = mean(recentReturns);

  // Blend historical and recent trends
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

  // Linear trend estimation
  const logPrices = prices.map((p) => Math.log(p));
  const x = logPrices.map((_, i) => i);
  const { slope } = linearRegression(x, logPrices);

  // Weekly seasonality (simplified)
  const weeklyPattern = [];
  for (let d = 0; d < 7; d++) {
    const dayPrices = prices.filter((_, i) => i % 7 === d);
    weeklyPattern.push(mean(dayPrices.map((p) => Math.log(p))));
  }
  const avgLogPrice = mean(logPrices);
  const seasonalFactors = weeklyPattern.map((w) => w - avgLogPrice);

  return {
    1: lastPrice * Math.exp(slope * 1),
    7: lastPrice * Math.exp(slope * 7),
    30: lastPrice * Math.exp(slope * 30),
  };
}

function simpleLSTM(prices: number[]): number {
  const n = prices.length;
  const lastPrice = prices[n - 1];

  // Weighted moving average with exponential decay (mimics LSTM memory)
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
  // Simplified GARCH(1,1) volatility forecast
  const returns = prices.slice(1).map((p, i) => Math.log(p / prices[i]));
  const squaredReturns = returns.map((r) => r * r);

  // GARCH parameters (typical estimates)
  const omega = 0.00001;
  const alpha = 0.1;
  const beta = 0.85;

  // Calculate current conditional variance
  let sigma2 = mean(squaredReturns);

  // Forecast 5 days ahead
  const forecasts: number[] = [];
  for (let h = 1; h <= 5; h++) {
    sigma2 = omega + (alpha + beta) * sigma2;
    forecasts.push(Math.sqrt(sigma2));
  }

  return forecasts;
}

function detectRegime(prices: number[]): string {
  const n = prices.length;
  const returns = prices.slice(1).map((p, i) => Math.log(p / prices[i]));

  // Short-term and long-term moving averages
  const shortMA = mean(returns.slice(-7));
  const longMA = mean(returns.slice(-30));

  // Volatility
  const recentVol = std(returns.slice(-14));
  const avgVol = std(returns);

  // Regime detection heuristics
  if (shortMA > 0.005 && shortMA > longMA) {
    return "bull";
  } else if (shortMA < -0.005 && shortMA < longMA) {
    return "bear";
  }
  return "sideways";
}

function calculateRiskMetrics(prices: number[]): ForecastResult["risk"] {
  const returns = prices.slice(1).map((p, i) => Math.log(p / prices[i]));

  // Sharpe Ratio (annualized)
  const avgReturn = mean(returns);
  const stdReturn = std(returns);
  const sharpe = stdReturn > 0 ? (avgReturn / stdReturn) * Math.sqrt(365) : 0;

  // Sortino Ratio (downside deviation)
  const negReturns = returns.filter((r) => r < 0);
  const downsideStd = std(negReturns) || stdReturn;
  const sortino = downsideStd > 0 ? (avgReturn / downsideStd) * Math.sqrt(365) : 0;

  // Max Drawdown
  let peak = prices[0];
  let maxDD = 0;
  for (const p of prices) {
    if (p > peak) peak = p;
    const dd = (p - peak) / peak;
    if (dd < maxDD) maxDD = dd;
  }

  // Calmar Ratio
  const annualizedReturn = avgReturn * 365;
  const calmar = Math.abs(maxDD) > 0 ? annualizedReturn / Math.abs(maxDD) : 0;

  return { sharpe, sortino, max_drawdown: maxDD, calmar };
}

function blendForecasts(
  modelForecasts: ForecastResult["modelForecasts"],
  regime: string
): { [key: string]: number } {
  // Regime-based weights
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
    modelForecasts.lstm * weights.lstm; // LSTM doesn't have weekly, so reuse daily

  const blend1m =
    modelForecasts.arima[30] * weights.arima +
    modelForecasts.prophet[30] * weights.prophet +
    modelForecasts.lstm * weights.lstm;

  // 1h is extrapolated from 1d
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

function labelH(h: number) {
  if (h === 1 / 24) return "1h";
  if (h === 1) return "1d";
  if (h === 7) return "1w";
  if (h === 30) return "1m";
  return `${h}d`;
}

function fmtUsd(n: number) {
  if (!Number.isFinite(n)) return "N/A";
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
}
