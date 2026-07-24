"""Testes da CausalSelfAttention (nb 04) — causalidade e KV-cache == recompute."""
from __future__ import annotations

import torch

from tucanoce.config import ModelConfig
from tucanoce.model.layers.attention import CausalSelfAttention
from tucanoce.model.layers.rope import precompute_rope_freqs


def _setup(embed_dim=16, n_heads=2, max_seq=32):
    cfg = ModelConfig(vocab_size=64, context_len=max_seq, n_layers=1,
                      n_heads=n_heads, embed_dim=embed_dim, dropout=0.0)
    attn = CausalSelfAttention(cfg).eval()
    cos, sin = precompute_rope_freqs(cfg.head_dim, max_seq, cfg.rope_base)
    return attn, cos, sin


def test_output_shape():
    attn, cos, sin = _setup()
    x = torch.randn(2, 10, 16)
    out, kv = attn(x, cos, sin)
    assert out.shape == x.shape and kv is None    # sem cache -> não retorna kv


def test_causal_mask_no_future_leak():
    # perturbar o ÚLTIMO token não pode mudar a saída da posição 0
    attn, cos, sin = _setup()
    x = torch.randn(1, 12, 16)
    x2 = x.clone()
    x2[0, -1] += 5.0
    y1, _ = attn(x, cos, sin)
    y2, _ = attn(x2, cos, sin)
    assert (y1[0, 0] - y2[0, 0]).abs().max() < 1e-6      # posição 0 não vê futuro
    assert (y1[0, -1] - y2[0, -1]).abs().max() > 1e-4    # última posição vê tudo


def test_use_cache_returns_kv():
    attn, cos, sin = _setup()
    x = torch.randn(1, 5, 16)
    _, kv = attn(x, cos, sin, use_cache=True)
    k, v = kv
    assert k.shape == (1, 2, 5, 8) and v.shape == (1, 2, 5, 8)  # (B,H,T,head_dim)


def test_kv_cache_equals_recompute():
    # decode token-a-token COM cache == recompute do forward inteiro SEM cache
    attn, cos, sin = _setup()
    seq = torch.randn(1, 10, 16)

    with torch.no_grad():
        outs_no = torch.stack(
            [attn(seq[:, :t], cos, sin)[0][:, -1] for t in range(1, 11)], dim=1)
        outs_c, past = [], None
        for t in range(10):
            y, past = attn(seq[:, t:t+1], cos, sin, past_kv=past, use_cache=True)
            outs_c.append(y[:, -1])
        outs_c = torch.stack(outs_c, dim=1)

    assert (outs_no - outs_c).abs().max() < 1e-4
