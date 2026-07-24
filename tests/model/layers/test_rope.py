"""Testes do RoPE (nb 04) — shapes, rotate_half e a INVARIÂNCIA relativa (a mágica)."""
from __future__ import annotations

import torch

from tucanoce.model.layers.rope import apply_rope, precompute_rope_freqs, rotate_half


def test_freqs_shapes():
    cos, sin = precompute_rope_freqs(head_dim=64, max_seq_len=128)
    assert cos.shape == (128, 64) and sin.shape == (128, 64)


def test_rotate_half():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    # [-x2, x1] com x1=[1,2], x2=[3,4] -> [-3,-4,1,2]
    assert torch.equal(rotate_half(x), torch.tensor([-3.0, -4.0, 1.0, 2.0]))


def test_apply_rope_at_pos0_is_identity():
    # posição 0: ângulo 0 -> cos=1, sin=0 -> rotação identidade
    cos, sin = precompute_rope_freqs(head_dim=8, max_seq_len=4)
    x = torch.randn(8)
    out = apply_rope(x.view(1, 8), cos[0:1], sin[0:1])
    assert torch.allclose(out.view(-1), x, atol=1e-6)


def test_rope_preserves_norm():
    # rotação preserva a norma do vetor
    cos, sin = precompute_rope_freqs(head_dim=16, max_seq_len=32)
    x = torch.randn(16)
    for pos in [1, 5, 20]:
        out = apply_rope(x.view(1, 16), cos[pos:pos+1], sin[pos:pos+1])
        assert torch.allclose(out.norm(), x.norm(), atol=1e-4)


def test_relative_invariance():
    # PROVA central: <R_m q, R_n k> depende SÓ de (n-m), não das posições absolutas
    head_dim = 64
    cos, sin = precompute_rope_freqs(head_dim, max_seq_len=64)
    q, k = torch.randn(head_dim), torch.randn(head_dim)

    def dot(m, n):
        qm = apply_rope(q.view(1, head_dim), cos[m:m+1], sin[m:m+1])
        kn = apply_rope(k.view(1, head_dim), cos[n:n+1], sin[n:n+1])
        return (qm * kn).sum().item()

    # mesma distância relativa (3) em posições absolutas diferentes -> mesmo produto
    assert abs(dot(2, 5) - dot(10, 13)) < 1e-4
    assert abs(dot(2, 5) - dot(40, 43)) < 1e-4
    # distância diferente -> produto diferente
    assert abs(dot(2, 5) - dot(2, 8)) > 1e-3


def test_base_controls_frequency():
    # base maior -> rotação mais lenta (comprimento de onda maior).
    # O índice 0 tem inv_freq=base^0=1 (invariante à base); testamos num índice >0.
    cos_small, _ = precompute_rope_freqs(8, 16, base=100.0)
    cos_large, _ = precompute_rope_freqs(8, 16, base=100000.0)
    # na última posição, base grande gira menos (cos mais próximo de 1)
    assert cos_large[-1, 2] > cos_small[-1, 2]
