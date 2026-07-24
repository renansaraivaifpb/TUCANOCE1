"""SwiGLU MLP — feed-forward estilo LLaMA (paper seção 5.2, Listing 2).

Substitui a MLP GELU-4d do GPT-2 por um Gated Linear Unit com ativação SiLU:
    SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
Três projeções (gate, up, down) em vez de duas, sem bias. hidden = 8d/3 arredondado
para múltiplo de 64 (mantém a contagem de parâmetros ~igual à MLP 4d).

Ver notebook 04_modernizacao_llama.ipynb.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import compute_hidden_dim


class SwiGLU(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = compute_hidden_dim(embed_dim)
        self.gate = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ramo com portão: SiLU(gate) filtra o ramo up (linear, sem distorção);
        # down projeta de volta a embed_dim. Dropout só no fim (§5.2).
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))
