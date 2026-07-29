"""Unit tests for scripts.train.

Smoke tests only — the real end-to-end training runs in scripts/scan.py. Here we
just want to prove that: (a) the loop actually reduces loss on a solvable toy,
(b) the seed leads to reproducible weights and scores.
"""
from __future__ import annotations

import numpy as np
import torch

from scripts.train import score_reconstruction, set_seed, train_autoencoder


def _toy_dataset(n=400, d=32, seed=0):
    """Structured data the AE can reconstruct: rank-4 gaussian latents embedded in ℝ^d."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 4))
    w = rng.standard_normal((4, d))
    x = z @ w + 0.05 * rng.standard_normal((n, d))
    return x.astype(np.float32)


def test_train_reduces_loss_on_toy():
    x = _toy_dataset(n=400, d=32, seed=0)
    result = train_autoencoder(
        x, batch_size=64, epochs=8, lr=1e-2, val_frac=0.2, seed=42,
    )
    # Baseline: reconstructing zeros ⇒ MSE ≈ variance of x ≈ 4/32 * some scale;
    # a trained AE should be materially below that.
    baseline = float((x ** 2).mean())
    assert result.final_val_loss < baseline * 0.7


def test_seed_reproducibility():
    """Two runs with the same seed and same tensor should yield identical scores."""
    x = _toy_dataset(n=300, d=16, seed=1)

    r1 = train_autoencoder(x, batch_size=32, epochs=3, val_frac=0.2, seed=123)
    score_x = x[:5]
    s1, _ = score_reconstruction(r1.model, score_x)

    r2 = train_autoencoder(x, batch_size=32, epochs=3, val_frac=0.2, seed=123)
    s2, _ = score_reconstruction(r2.model, score_x)

    np.testing.assert_allclose(s1, s2, rtol=1e-5, atol=1e-6)


def test_early_stopping_triggers_on_static_val():
    """When val loss plateaus we should stop before max epochs."""
    # Very small dataset ⇒ val loss stabilizes quickly.
    x = _toy_dataset(n=40, d=8, seed=2)
    r = train_autoencoder(x, batch_size=16, epochs=200, lr=1e-3,
                          val_frac=0.25, early_stop_patience=2, seed=7)
    assert r.n_epochs_ran < 200


def test_score_reconstruction_shape():
    x = _toy_dataset(n=100, d=16, seed=3)
    r = train_autoencoder(x, batch_size=32, epochs=1, seed=0)
    score = x[:7]
    per_sample, per_col_sq = score_reconstruction(r.model, score)
    assert per_sample.shape == (7,)
    assert per_col_sq.shape == (7, 16)
    # per_sample should be the mean of per_col_sq along axis 1.
    np.testing.assert_allclose(per_sample, per_col_sq.mean(axis=1), rtol=1e-5, atol=1e-7)


def test_set_seed_makes_torch_randn_deterministic():
    set_seed(0)
    a = torch.randn(3, 3)
    set_seed(0)
    b = torch.randn(3, 3)
    torch.testing.assert_close(a, b)
