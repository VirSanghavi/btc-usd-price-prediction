"use client";

import { useState } from "react";

type ForecastResult = {
  lastPrice: number;
  mu: number;
  sigma: number;
  forecasts: { h: number; price: number }[];
  hitProb: { h: number; prob: number }[];
};

const HORIZONS = [1 / 24, 1, 7, 30]; // 1h, 1d, 1w, 1m in days
const TARGET = 99_000;

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

      setResult({ lastPrice, mu, sigma, forecasts, hitProb });
    } catch (e: any) {
      setError(e?.message || "Failed to compute forecast");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel" style={{ padding: 16, marginTop: 8 }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 8 }}>
        <button className="button buttonPrimary" onClick={onClick} disabled={loading}>
          {loading ? "Computing forecast…" : "Get Forecast (Browser)"}
        </button>
        {error && <span style={{ color: "#fca5a5", fontSize: 13 }}>{error}</span>}
      </div>
      {result && (
        <div style={{ fontSize: 13, color: "#e5e7eb", display: "grid", gap: 6 }}>
          <div style={{ color: "#9ca3af" }}>
            Using last 1 year of BTC prices (CoinGecko) and a simple geometric-Brownian model.
          </div>
          <div>
            <strong>Last price:</strong> {fmtUsd(result.lastPrice)}
          </div>
          <div>
            <strong>Est. daily drift μ:</strong> {(result.mu * 100).toFixed(2)}% &nbsp;·&nbsp;
            <strong>daily volatility σ:</strong> {(result.sigma * 100).toFixed(2)}%
          </div>
          <div>
            <strong>Forecasts:</strong>
            {" "}
            {result.forecasts.map((f, idx) => (
              <span key={f.h} style={{ marginRight: 12 }}>
                {labelH(f.h)}: {fmtUsd(f.price)}
                {idx < result.forecasts.length - 1 && " | "}
              </span>
            ))}
          </div>
          <div>
            <strong>Prob. BTC hits $99k within horizon:</strong>
            {" "}
            {result.hitProb.map((p, idx) => (
              <span key={p.h} style={{ marginRight: 12 }}>
                {labelH(p.h)}: {(p.prob * 100).toFixed(2)}%
                {idx < result.hitProb.length - 1 && " | "}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
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
    const steps = Math.max(1, Math.round(h * 24)); // roughly hourly steps
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

// Box–Muller transform
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
