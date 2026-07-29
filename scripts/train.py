"""Training loop for the MLP autoencoder.

Design §:
  - Loss:     MSE (elementwise).
  - Optim:    Adam, lr=1e-3.
  - Batch:    256.
  - Epochs:   max 50, early stop patience 5 on the held-out validation MSE.
  - Split:    80/20 random train/val, seeded by --seed.
  - Device:   auto-detects cuda / mps / cpu (falls back to cpu when neither exists).
  - Seed:     np.random + torch.manual_seed for reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.model import Autoencoder


@dataclass
class TrainResult:
    model: nn.Module
    final_train_loss: float
    final_val_loss: float
    n_epochs_ran: int
    device: str


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_train_val(
    x: np.ndarray,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    cut = int(round(x.shape[0] * (1 - val_frac)))
    return x[idx[:cut]], x[idx[cut:]]


def train_autoencoder(
    train_x: np.ndarray,
    *,
    input_dim: int | None = None,
    batch_size: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    early_stop_patience: int = 5,
    seed: int = 42,
    device: torch.device | None = None,
) -> TrainResult:
    """Train an Autoencoder on `train_x` and return the fitted model + diagnostics.

    Args:
        train_x: (N, D) float32 z-scored samples. If D is small enough for CPU (~160),
                 training completes in seconds on a laptop.

    Deterministic under a fixed `seed` when the same tensors are provided.
    """
    if train_x.shape[0] < 4:
        raise ValueError(
            f"train_x has only {train_x.shape[0]} rows — need at least 4 for a train/val split."
        )

    set_seed(seed)
    device = device or _pick_device()

    d = input_dim if input_dim is not None else train_x.shape[1]
    model = Autoencoder(input_dim=d).to(device)

    tr_x, va_x = _split_train_val(train_x.astype(np.float32), val_frac, seed)
    tr_t = torch.from_numpy(tr_x)
    va_t = torch.from_numpy(va_x)

    train_loader = DataLoader(
        TensorDataset(tr_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state: dict | None = None
    since_improve = 0
    ran = 0
    last_train_loss = float("nan")

    for epoch in range(epochs):
        ran = epoch + 1
        model.train()
        loss_sum = 0.0
        n = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * batch.shape[0]
            n += batch.shape[0]
        last_train_loss = loss_sum / max(n, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            va_batch = va_t.to(device)
            if va_batch.numel() == 0:
                val_loss = last_train_loss
            else:
                out = model(va_batch)
                val_loss = loss_fn(out, va_batch).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(
        model=model.eval(),
        final_train_loss=float(last_train_loss),
        final_val_loss=float(best_val),
        n_epochs_ran=int(ran),
        device=str(device),
    )


def score_reconstruction(
    model: nn.Module,
    score_x: np.ndarray,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample reconstruction MSE and per-feature-column squared error.

    Returns:
        per_sample_mse: (N,)  — mean squared error per row (bulk anomaly score).
        per_col_sq:      (N, D) — squared error per feature column (used to derive
                                 top_feature attribution downstream).
    """
    device = device or _pick_device()
    model = model.to(device).eval()
    with torch.no_grad():
        x = torch.from_numpy(score_x.astype(np.float32)).to(device)
        out = model(x)
        sq = (out - x).pow(2).cpu().numpy()
    per_sample = sq.mean(axis=1)
    return per_sample, sq
