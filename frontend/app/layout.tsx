import React from "react";

export const metadata = {
  title: "BTC-USD Price Prediction",
  description: "Mr. Tristan; use this to estimate when BTC-USD will hit $99K: your target price!"
};

import "./globals.css";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
