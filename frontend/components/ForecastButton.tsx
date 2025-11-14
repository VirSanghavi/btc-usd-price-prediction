"use client";

import { useState } from "react";

type Forecast = {
  ok: boolean;
  last_price?: number;
  realtime_price?: number | null;
  risk?: Record<string, number>;
  regime?: string;
  blended?: Record<string, number>;
  monte_carlo?: Record<string, number> | Record<number, number>;
  exit_signal?: { signal: boolean; score: number; explanation: string };
  error?: string;
};

export default function ForecastButton() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<Forecast | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function getForecast() {
    try {
      setLoading(true);
      setErr(null);
      setData(null);
      const base = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
      const res = await fetch(`${base}/api/forecast`, { cache: "no-store" });
      const json: Forecast = await res.json();
      if (!json.ok) throw new Error(json.error || "Failed");
      setData(json);
    } catch (e: any) {
      setErr(e?.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel" style={{ padding: 16 }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <button className="button buttonPrimary" onClick={getForecast} disabled={loading}>
          {loading ? "Fetching…" : "Get Forecast"}
        </button>
        {err && <span style={{ color: "#fca5a5" }}>{err}</span>}
      </div>
      {data && (
        <div style={{ marginTop: 12, display: "grid", gap: 8 }}>
          <div style={{ color: "#9ca3af", fontSize: 13 }}>Regime: {data.regime || "unknown"}</div>
          <div>
            <strong>Blended</strong>: {data.blended ? (
              <>
                <span style={{ marginLeft: 8 }}>1h: {fmt(data.blended["1h"])}</span>
                <span style={{ marginLeft: 8 }}>1d: {fmt(data.blended["1d"])}</span>
                <span style={{ marginLeft: 8 }}>1w: {fmt(data.blended["1w"])}</span>
                <span style={{ marginLeft: 8 }}>1m: {fmt(data.blended["1m"])}</span>
              </>
            ) : (
              "N/A"
            )}
          </div>
          {data.monte_carlo && (
            <div style={{ color: "#9ca3af", fontSize: 13 }}>
              Prob. hit $99k (1d/7d/30d): {pct(data.monte_carlo[1 as any])} / {pct(data.monte_carlo[7 as any])} / {pct(data.monte_carlo[30 as any])}
            </div>
          )}
          {data.exit_signal && (
            <div>
              <strong>Exit</strong>: {data.exit_signal.signal ? "EXIT" : "HOLD"} (score {data.exit_signal.score.toFixed(2)})
              <div style={{ color: "#9ca3af", fontSize: 13, marginTop: 4 }}>{data.exit_signal.explanation}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function fmt(n?: number) {
  if (typeof n !== "number" || !isFinite(n)) return "N/A";
  return `$${n.toLocaleString()}`;
}

function pct(n?: number) {
  if (typeof n !== "number" || !isFinite(n)) return "N/A";
  return `${(n * 100).toFixed(2)}%`;
}
