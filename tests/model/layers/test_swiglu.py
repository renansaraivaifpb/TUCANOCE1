"""Testes do SwiGLU (nb 04) — shape, ausência de bias, paridade de params com GELU-4d."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tucanoce.config import compute_hidden_dim
from tucanoce.model.layers.swiglu import SwiGLU


def test_shape_preserved():
    mlp = SwiGLU(32, dropout=0.0)
    x = torch.randn(2, 10, 32)
    assert mlp(x).shape == x.shape


def test_no_bias():
    mlp = SwiGLU(32)
    for proj in (mlp.gate, mlp.up, mlp.down):
        assert proj.bias is None


def test_param_count_matches_gelu_4d():
    # o ponto de h = 8d/3: SwiGLU (3 projeções) ≈ MLP GELU-4d (2 projeções) em params
    d = 512
    swiglu = SwiGLU(d, dropout=0.0)
    gelu_mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    n_swi = sum(p.numel() for p in swiglu.parameters())
    n_gelu = sum(p.numel() for p in gelu_mlp.parameters())
    assert abs(n_swi / n_gelu - 1.0) < 0.05     # ~1x por construção


def test_hidden_dim_default():
    d = 256
    mlp = SwiGLU(d)
    assert mlp.gate.out_features == compute_hidden_dim(d)


def test_matches_manual_formula():
    # SwiGLU(x) = down(silu(gate(x)) * up(x))
    d = 16
    mlp = SwiGLU(d, dropout=0.0).eval()
    x = torch.randn(2, d)
    expected = mlp.down(F.silu(mlp.gate(x)) * mlp.up(x))
    assert torch.allclose(mlp(x), expected, atol=1e-6)
