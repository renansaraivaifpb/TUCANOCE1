"""Modelo completo: transformer decoder-only (nb 03).

    h0     = Embed(x)
    h_l    = bloco_l(h_{l-1})   para l = 1..L
    logits = Norm(h_L) @ W_e^T   (weight tying — nb 03)

Inicialização (nb 03): Linear ~ N(0, 0.02^2); projeções residuais (out_proj
da atenção e down da MLP) escaladas por 1/sqrt(2L) para manter a norma O(1).

forward(x, past_kvs=None, use_cache=False):
    - sem cache (treino/eval): retorna logits
    - com cache (inferência): retorna (logits, new_past_kvs)  — nb 04

Ver notebooks 03_arquitetura_base.ipynb (base) e 04 (modernização).
"""
from __future__ import annotations

import math

import torch.nn as nn

from ..config import ModelConfig
from .block import Block
from .layers.rmsnorm import RMSNorm
from .layers.rope import precompute_rope_freqs


class TucanoCE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.embed_dim)
        # weight tying: lm_head compartilha pesos com a embedding
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        # cos/sin de RoPE pré-computados como buffers (não são parâmetros treináveis;
        # persistent=False p/ não inflar o checkpoint — são deriváveis do cfg).
        cos, sin = precompute_rope_freqs(cfg.head_dim, cfg.context_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._init_weights()

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_weights(self):
        # Passada 1: todo Linear/Embedding ~ N(0, 0.02^2), biases zerados.
        self.apply(self._init_module)
        # Passada 2: projeções residuais (out_proj da atenção e down da SwiGLU)
        # recebem escala 1/sqrt(2L) — cancela o crescimento ~sqrt(2L) da norma da
        # rodovia residual (nb 03). São exatamente os pesos SOMADOS de volta em h.
        std_res = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=std_res)

    def forward(self, x, past_kvs=None, use_cache=False):
        h = self.embed(x)
        cos, sin = self.rope_cos, self.rope_sin
        new_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            h, kv = block(h, cos, sin, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                new_kvs.append(kv)
        h = self.norm_f(h)
        logits = self.lm_head(h)
        if use_cache:
            return logits, new_kvs
        return logits

    def num_params(self) -> int:
        # nn.Module.parameters() já deduplica pesos compartilhados: com weight
        # tying, embed.weight/lm_head.weight é o MESMO tensor e aparece uma vez só.
        # Basta somar — subtrair V*d aqui removeria a embedding da conta (nb 03).
        return sum(p.numel() for p in self.parameters())
