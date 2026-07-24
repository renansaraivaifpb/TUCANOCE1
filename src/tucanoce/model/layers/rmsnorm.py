"""RMSNorm — normalização estilo LLaMA (paper seção 5.1, Listing 1).

Substitui LayerNorm: descarta a re-centralização (subtrair a média) e o bias beta,
mantendo só a divisão pelo RMS. Geometricamente projeta x na esfera de raio sqrt(d)
antes da escala gamma — preserva direção, descarta magnitude.

Cuidado de precisão: a soma de quadrados é feita em FP32 mesmo sob BF16 (seção 5.1.6).

Ver derivação e implementação no notebook 04_modernizacao_llama.ipynb.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS calculado em FP32 mesmo sob BF16: a soma de d quadrados acumula erro
        # em precisão reduzida (§5.1.6). Voltamos ao dtype de entrada só no fim.
        dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        out = (x_fp32 * rms).to(dtype)
        return out * self.weight
