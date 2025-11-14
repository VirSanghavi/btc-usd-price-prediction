# btc-usd-price-prediction
predict btc recovery time to $99k!

## Frontend (Next.js)

A simple Next.js landing page with a live BTC/USD price ticker is available in `frontend/`.

### Quick start

Requires Node.js 18+.

```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:3000 in your browser.

### Build and run

```bash
cd frontend
npm run build
npm start
```

This frontend is independent of the Python script. It now includes a "Get Forecast" button that calls the backend API below.

## Backend API (FastAPI)

Start the FastAPI server which wraps the `crypto_quant` pipeline:

```bash
pip install -r requirements.txt
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Optional env vars:
- `GLASSNODE_API_KEY` (on-chain metrics)
- `FRED_API_KEY` (US interest rates)
- `BTC_HIST_CSV` (custom daily BTC CSV)

With the server running, open the frontend at http://localhost:3000 and click "Get Forecast".

On Vercel, set the environment variable `NEXT_PUBLIC_API_BASE` to your backend base URL (e.g., `https://your-backend.example.com`). The frontend will call `${NEXT_PUBLIC_API_BASE}/api/forecast`.
