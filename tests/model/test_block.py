"""Testes do Block pre-norm (nb 03) — contrato de forma e caminho residual."""
from __future__ import annotations

import torch

from tucanoce.model.block import Block
from tucanoce.model.layers.rope import precompute_rope_freqs


def _block(tiny_cfg):
    blk = Block(tiny_cfg).eval()
    cos, sin = precompute_rope_freqs(tiny_cfg.head_dim, tiny_cfg.context_len,
                                     tiny_cfg.rope_base)
    return blk, cos, sin


def test_returns_tuple_and_shape(tiny_cfg):
    blk, cos, sin = _block(tiny_cfg)
    x = torch.randn(2, 8, tiny_cfg.embed_dim)
    out, kv = blk(x, cos, sin)
    assert out.shape == x.shape and kv is None


def test_use_cache_returns_kv(tiny_cfg):
    blk, cos, sin = _block(tiny_cfg)
    x = torch.randn(1, 6, tiny_cfg.embed_dim)
    _, kv = blk(x, cos, sin, use_cache=True)
    assert kv is not None and len(kv) == 2      # (k, v)


def test_residual_path(tiny_cfg):
    # se atenção e MLP retornassem ~0, a saída seria ~a entrada (rodovia residual).
    # aqui checamos só que a saída NÃO é uma transformação que destrói a entrada:
    # zerando os pesos de saída dos dois ramos, out deve ser exatamente x.
    blk, cos, sin = _block(tiny_cfg)
    with torch.no_grad():
        blk.attn.out_proj.weight.zero_()
        blk.mlp.down.weight.zero_()
    x = torch.randn(1, 5, tiny_cfg.embed_dim)
    out, _ = blk(x, cos, sin)
    assert torch.allclose(out, x, atol=1e-6)    # h = h + 0 + 0
