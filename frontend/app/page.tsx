"use client";
import Link from "next/link";
import PriceTicker from "@/components/PriceTicker";
import BrowserForecast from "@/components/BrowserForecast";

export default function Page() {
  return (
    <main>
      <div className="container" style={{ paddingTop: 56, paddingBottom: 24 }}>
        <section className="panel hero">
          <div>
            <div className="title">BTC-USD Price Prediction</div>
            <p className="subtitle">
              built by vir sanghavi
            </p>
          </div>

          <div className="ctaRow">
            <Link className="button buttonPrimary" href="https://github.com/VirSanghavi/btc-usd-price-prediction" target="_blank">
              View Repository
            </Link>
            <Link className="button" href="https://www.coingecko.com/en/coins/bitcoin" target="_blank">
              Bitcoin Overview
            </Link>
          </div>
        </section>

        <div style={{ height: 16 }} />

        <PriceTicker />

        <BrowserForecast />

        <p className="footer">
          All forecasts are computed client-side from live BTC data (CoinGecko).
        </p>
      </div>
    </main>
  );
}
