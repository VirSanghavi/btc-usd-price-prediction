"""PyTorch LSTM model for BTC log returns."""
from __future__ import annotations

import numpy as np
import pandas as pd


def lstm_predict_next_prices(df: pd.DataFrame, lookback: int = 30, epochs: int = 3, hidden: int = 32) -> dict:
    """Train a small LSTM on log returns and predict next prices.

    Args:
        df (pd.DataFrame): Daily data with 'date' and 'close' or 'price'.
        lookback (int): Sequence length.
        epochs (int): Training epochs (kept small for speed).
        hidden (int): Hidden size.

    Returns:
        dict: {'next_return': float, 'next_price': float}
    """

    try:
        import torch
        import torch.nn as nn

        data = df.copy()
        if "close" not in data.columns and "price" in data.columns:
            data["close"] = data["price"]
        y = data["close"].dropna().astype(float).values
        if len(y) < lookback + 5:
            return {"next_return": float("nan"), "next_price": float("nan")}
        logp = np.log(y + 1e-9)
        rets = np.diff(logp)
        X = []
        Y = []
        for i in range(len(rets) - lookback):
            X.append(rets[i : i + lookback])
            Y.append(rets[i + lookback])
        X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
        Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(-1)

        class LSTM(nn.Module):
            def __init__(self, hidden: int):
                super().__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        model = LSTM(hidden)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            opt.step()

        model.eval()
        last_seq = torch.tensor(rets[-lookback:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        next_ret = float(model(last_seq).detach().numpy().ravel()[0])
        next_price = float(np.exp(np.log(y[-1]) + next_ret))
        return {"next_return": next_ret, "next_price": next_price}
    except Exception:
        return {"next_return": float("nan"), "next_price": float("nan")}
