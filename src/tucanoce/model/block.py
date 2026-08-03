"""Bloco transformer pre-norm (nb 03).

    h = h + Attn(Norm(h))
    h = h + MLP(Norm(h))

Pre-norm (norma ANTES da atenção/MLP) estabiliza o treino em modelos profundos;
post-norm (paper original) tende a explodir gradiente em > 10 camadas.

Ver notebook 03_arquitetura_base.ipynb.
"""
from __future__ import annotations

import torch.nn as nn

from ..config import ModelConfig
from .layers.attention import CausalSelfAttention
from .layers.rmsnorm import RMSNorm
from .layers.swiglu import SwiGLU


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.embed_dim)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.embed_dim)
        self.mlp = SwiGLU(cfg.embed_dim, cfg.hidden_dim, cfg.dropout)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        # Pre-norm: norma ANTES de cada estágio, soma na rodovia residual limpa.
        attn_out, new_kv = self.attn(self.norm1(x), cos, sin,
                                     past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv
