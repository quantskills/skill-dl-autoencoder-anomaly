"""Unit tests for scripts.model.Autoencoder."""
from __future__ import annotations

import torch

from scripts.model import Autoencoder, count_parameters


def test_forward_shape_matches_input_shape():
    model = Autoencoder(input_dim=160)
    x = torch.randn(4, 160)
    out = model(x)
    assert out.shape == (4, 160)


def test_encoder_bottleneck_dim():
    model = Autoencoder(input_dim=160, hidden=(96, 48), code_dim=32)
    x = torch.randn(4, 160)
    z = model.encode(x)
    assert z.shape == (4, 32)


def test_param_count_in_expected_range():
    """160→96→48→32→48→96→160 with biases: ~ (160+1)*96 + (96+1)*48 + (48+1)*32 doubled = ~44k."""
    model = Autoencoder(input_dim=160, hidden=(96, 48), code_dim=32)
    n = count_parameters(model)
    assert 30_000 <= n <= 80_000, f"unexpected param count {n}"


def test_final_decoder_layer_has_no_relu():
    """The last decoder Module should be a bare Linear so outputs are unbounded."""
    model = Autoencoder(input_dim=160)
    last = list(model.decoder.children())[-1]
    assert isinstance(last, torch.nn.Linear)


def test_dropout_disabled_in_eval_makes_forward_deterministic():
    torch.manual_seed(0)
    model = Autoencoder(input_dim=160).eval()
    x = torch.randn(2, 160)
    a = model(x)
    b = model(x)
    torch.testing.assert_close(a, b)
