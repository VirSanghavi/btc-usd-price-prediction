"use client";

import { useEffect, useState } from "react";

type PriceData = {
  usd: number;
};

export default function PriceTicker() {
  const [price, setPrice] = useState<number | null>(null);
  const [updated, setUpdated] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  async function fetchPrice() {
    try {
      setError(null);
      const res = await fetch(
        "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
        { cache: "no-store" }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = (await res.json()) as { bitcoin?: PriceData };
      const v = json.bitcoin?.usd ?? null;
      setPrice(v);
      setUpdated(new Date().toLocaleTimeString());
    } catch (e: any) {
      setError(e?.message || "Failed to load price");
    }
  }

  useEffect(() => {
    fetchPrice();
    const id = setInterval(fetchPrice, 30_000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="panel ticker">
      <div>
        <div style={{ fontSize: 12, color: "#9ca3af" }}>BTC/USD</div>
        <div style={{ fontSize: 22, fontWeight: 700 }}>
          {price !== null ? `$${price.toLocaleString()}` : "Loading..."}
        </div>
      </div>
      <div style={{ textAlign: "right" }}>
        <div style={{ fontSize: 12, color: "#9ca3af" }}>Updated</div>
        <div style={{ fontSize: 14 }}>{updated || "—"}</div>
        {error && (
          <div style={{ color: "#fca5a5", fontSize: 12, marginTop: 4 }}>{error}</div>
        )}
      </div>
    </div>
  );
}
