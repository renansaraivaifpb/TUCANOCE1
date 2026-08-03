"""RoPE — Rotary Position Embeddings (nb 04).

Em vez de somar embeddings posicionais (GPT-2), rotaciona Q e K por um ângulo
dependente da posição. A pontuação de atenção passa a depender só da posição
RELATIVA (n - m), o que dá robustez a extrapolação de contexto.

frequências: theta_i = base^(-2i/head_dim), base=10000 (nb 04).

Ver derivação (invariância relativa) no notebook 04_modernizacao_llama.ipynb.
"""
from __future__ import annotations

import torch


def precompute_rope_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """Tabelas cos/sin de shape (max_seq_len, head_dim), no formato "rotate_half".

    theta_i = base^(-2i/head_dim): dimensões baixas giram rápido (ordem local),
    altas giram devagar (contexto distante). Duplicamos freqs ([f, f]) para casar
    com o rotate_half — mais simples em código que a rotação por pares interleaved.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, inv_freq)       # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)         # (T, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # [-x2, x1]: implementa a multiplicação pela parte imaginária da rotação 2D.
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (..., T, head_dim); cos/sin: (T, head_dim). x*cos + rotate_half(x)*sin
    # é exatamente R(theta)·x aplicado em cada par de dimensões.
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return (x * cos) + (rotate_half(x) * sin)
