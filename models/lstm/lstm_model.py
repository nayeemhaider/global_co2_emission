from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Simple per-series LSTM 

class LSTMForecaster(nn.Module):
    """
    Simple univariate LSTM for time-series forecasting.
    Input: sequence of past emissions
    Output: next-step emission prediction
    """

    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # last timestep
        out = self.fc(out)
        return out


def create_sequences(values: np.ndarray, seq_len: int = 5):
    """
    values: 1D numpy array of emissions (already scaled if needed)
    returns X (N, seq_len, 1), y (N, 1) as torch tensors
    """
    xs, ys = [], []
    for i in range(len(values) - seq_len):
        x = values[i : i + seq_len]
        y = values[i + seq_len]
        xs.append(x)
        ys.append(y)
    if not xs:
        return torch.empty(0, seq_len, 1), torch.empty(0, 1)
    X = torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1)
    return X, y

# Global LSTM with country embeddings

class GlobalLSTMForecaster(nn.Module):
    """
    Global LSTM model that learns from all countries jointly.

    For each time step:
      input = [emission_t (scaled), country_embedding]
    """

    def __init__(
        self,
        n_countries: int,
        emb_dim: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_countries = n_countries
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.country_emb = nn.Embedding(n_countries, emb_dim)
        self.lstm = nn.LSTM(
            input_size=1 + emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, country_idx):
        """
        x: (batch, seq_len, 1)  -- scaled emissions
        country_idx: (batch,)   -- int indices for country
        """
        emb = self.country_emb(country_idx)              # (batch, emb_dim)
        emb_seq = emb.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch, seq_len, emb_dim)
        x_cat = torch.cat([x, emb_seq], dim=-1)          # (batch, seq_len, 1 + emb_dim)

        out, _ = self.lstm(x_cat)
        out_last = out[:, -1, :]                         # (batch, hidden_size)
        out = self.fc(out_last)                          # (batch, 1)
        return out


# Save / load helpers for global LSTM

def save_global_lstm(
    model: GlobalLSTMForecaster,
    mean: float,
    std: float,
    country_to_idx: Dict[str, int],
    path: Path,
):
    """
    Save model + scaling stats + country mapping.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "state_dict": model.state_dict(),
        "mean": float(mean),
        "std": float(std),
        "country_to_idx": country_to_idx,
        "n_countries": model.n_countries,
        "emb_dim": model.emb_dim,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
    }
    torch.save(obj, path)


def load_global_lstm(path: Path) -> Tuple[GlobalLSTMForecaster, float, float, Dict[str, int]]:
    """
    Load global LSTM model and metadata.
    """
    obj = torch.load(path, map_location="cpu")
    n_countries = obj["n_countries"]
    emb_dim = obj["emb_dim"]
    hidden_size = obj["hidden_size"]
    num_layers = obj["num_layers"]
    dropout = obj["dropout"]

    model = GlobalLSTMForecaster(
        n_countries=n_countries,
        emb_dim=emb_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.load_state_dict(obj["state_dict"])
    model.eval()

    mean = obj["mean"]
    std = obj["std"]
    country_to_idx = obj["country_to_idx"]
    return model, mean, std, country_to_idx


# Forecast helper for a single country using global LSTM

def global_lstm_forecast_country(
    series: np.ndarray,
    country: str,
    model: GlobalLSTMForecaster,
    mean: float,
    std: float,
    country_to_idx: Dict[str, int],
    horizon: int = 10,
    seq_len: int = 5,
) -> np.ndarray:
    """
    series: 1D numpy array of historical emissions (unscaled, full history)
    country: country name (must be in country_to_idx)
    horizon: number of steps to forecast
    Returns: unscaled forecast as numpy array
    """
    if country not in country_to_idx:
        raise ValueError(f"Country '{country}' not found in country_to_idx mapping.")

    country_id = country_to_idx[country]
    country_tensor = torch.tensor([country_id], dtype=torch.long)

    values = series.astype(float)
    scaled = (values - mean) / std

    history = scaled.copy()
    preds_scaled = []

    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            if len(history) < seq_len:
                seq = np.pad(history, (seq_len - len(history), 0), mode="edge")
            else:
                seq = history[-seq_len:]
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)

            y_hat_scaled = model(x, country_tensor).item()
            preds_scaled.append(y_hat_scaled)
            history = np.append(history, y_hat_scaled)

    preds_scaled = np.array(preds_scaled)
    preds = preds_scaled * std + mean
    return preds
