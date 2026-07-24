"""Atenção causal multi-head com RoPE e KV-cache (paper seções 4.2 e 5.4).

- Projeta h em Q, K, V; quebra em H cabeças de dimensão head_dim = d/H.
- Aplica RoPE em Q e K antes do produto interno.
- Usa F.scaled_dot_product_attention (Flash Attention quando o hardware suporta),
  evitando materializar a matriz QK^T de tamanho T x T.
- No decode com cache, is_causal vira Q.size(-2) == K.size(-2) (Listing 4).

Ver notebooks 03_arquitetura_base.ipynb e 04_modernizacao_llama.ipynb.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import ModelConfig
from .rope import apply_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.k_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.v_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.out_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        B, T, C = x.shape
        H, hd = self.n_heads, self.head_dim
        # projeta e quebra em cabeças: (B, H, T, hd)
        q = self.q_proj(x).view(B, T, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, hd).transpose(1, 2)

        # offset posicional: com cache, o token novo está na posição global p+k,
        # então usamos cos/sin dessa posição, não da posição 0 (Eq. 54).
        offset = 0 if past_kv is None else past_kv[0].size(-2)
        c = cos[offset:offset + T]
        s = sin[offset:offset + T]
        q = apply_rope(q, c, s)
        k = apply_rope(k, c, s)

        # concatena K,V ao cache (já trazem a rotação das posições originais)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=-2)
            v = torch.cat([past_kv[1], v], dim=-2)
        new_kv = (k, v) if use_cache else None

        # is_causal adaptativo: True no prefill (|Q|==|K|), False no decode de 1 token
        # (a query única "vê" todo o cache sem máscara). Listing 4.
        is_causal = q.size(-2) == k.size(-2)
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), new_kv
