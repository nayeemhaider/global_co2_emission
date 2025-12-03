from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data_cleaning import prepare_core_frame
from src.config import LSTM_MODELS_DIR
from models.lstm.lstm_model import GlobalLSTMForecaster, save_global_lstm


def build_global_lstm_dataset(
    val_years_per_country: int = 5,
    seq_len: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Build train/val dataset for global LSTM from all countries.

    For each country:
      - sort by year
      - build rolling sequences of length `seq_len`
      - target = next emission
      - samples whose target year falls in last `val_years_per_country` years
        are assigned to validation; others to training.
    """
    df = prepare_core_frame()

    # Map country -> index
    countries = sorted(df["country"].unique())
    country_to_idx = {c: i for i, c in enumerate(countries)}

    X_train_list: List[np.ndarray] = []
    y_train_list: List[float] = []
    c_train_list: List[int] = []

    X_val_list: List[np.ndarray] = []
    y_val_list: List[float] = []
    c_val_list: List[int] = []

    for country in countries:
        sub = df[df["country"] == country].copy()
        sub = sub.sort_values("year")
        values = sub["emission"].values.astype(float)
        years = sub["year"].values.astype(int)

        if len(values) <= seq_len + 1:
            continue

        # Determine validation cut based on years
        unique_years = np.unique(years)
        if len(unique_years) <= val_years_per_country + 1:
            # Not enough years to meaningfully hold out; skip or put all in train
            val_cut_year = unique_years[-1] + 1  # future year -> no val
        else:
            val_cut_year = unique_years[-val_years_per_country]

        c_idx = country_to_idx[country]

        # Build sequences
        for i in range(len(values) - seq_len):
            x_seq = values[i : i + seq_len]
            y_target = values[i + seq_len]
            target_year = years[i + seq_len]

            if target_year >= val_cut_year:
                X_val_list.append(x_seq)
                y_val_list.append(y_target)
                c_val_list.append(c_idx)
            else:
                X_train_list.append(x_seq)
                y_train_list.append(y_target)
                c_train_list.append(c_idx)

    if not X_train_list or not X_val_list:
        raise RuntimeError("Not enough data to build train/val sets for global LSTM.")

    X_train = np.stack(X_train_list)  # (N_train, seq_len)
    y_train = np.array(y_train_list)  # (N_train,)
    c_train = np.array(c_train_list, dtype=int)

    X_val = np.stack(X_val_list)      # (N_val, seq_len)
    y_val = np.array(y_val_list)
    c_val = np.array(c_val_list, dtype=int)

    return X_train, y_train, X_val, y_val, c_train, c_val, country_to_idx


def train_global_lstm(
    seq_len: int = 5,
    val_years_per_country: int = 5,
    hidden_size: int = 64,
    emb_dim: int = 8,
    num_layers: int = 2,
    dropout: float = 0.2,
    max_epochs: int = 400,
    patience: int = 30,
    lr: float = 1e-3,
):
    print("Building global LSTM dataset...")
    (
        X_train_raw,
        y_train_raw,
        X_val_raw,
        y_val_raw,
        c_train,
        c_val,
        country_to_idx,
    ) = build_global_lstm_dataset(
        val_years_per_country=val_years_per_country,
        seq_len=seq_len,
    )

    # Standardize using train only
    all_train_vals = X_train_raw.reshape(-1)
    mean = float(all_train_vals.mean())
    std = float(all_train_vals.std())
    if std == 0:
        std = 1.0

    X_train = (X_train_raw - mean) / std
    y_train = (y_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std
    y_val = (y_val_raw - mean) / std

    # Convert to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # (N, seq_len, 1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)  # (N, 1)
    c_train_t = torch.tensor(c_train, dtype=torch.long)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
    c_val_t = torch.tensor(c_val, dtype=torch.long)

    n_countries = len(country_to_idx)
    print(f"Training global LSTM on {len(X_train)} train samples, {len(X_val)} val samples, {n_countries} countries.")

    model = GlobalLSTMForecaster(
        n_countries=n_countries,
        emb_dim=emb_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        pred_train = model(X_train_t, c_train_t)
        loss_train = criterion(pred_train, y_train_t)
        loss_train.backward()
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_t, c_val_t)
            loss_val = criterion(pred_val, y_val_t).item()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{max_epochs} - Train loss: {loss_train.item():.6f} - Val loss: {loss_val:.6f}")

        if loss_val < best_val_loss - 1e-6:
            best_val_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save global model
    out_path = LSTM_MODELS_DIR / "global_lstm.pt"
    save_global_lstm(model, mean, std, country_to_idx, out_path)
    print(f"Global LSTM saved to: {out_path}")
    print(f"Global scaling: mean={mean:.3f}, std={std:.3f}")


if __name__ == "__main__":
    train_global_lstm()
