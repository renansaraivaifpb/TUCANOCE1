"""Avaliação (nb 07).

Métricas:
- val_loss (cross-entropy em nats/token) — o que o early stopping monitora.
- next-token accuracy.
- bits per character (BPC): L_val / (bytes_por_token * ln 2). Comparável entre
  tokenizers diferentes, ao contrário da cross-entropy crua.

Interpretação: o piso trivial sem modelo é log(V). A redução mede informação
ganha por token.

Ver notebook 07_avaliacao_scaling.ipynb.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, val_loader, device=None) -> dict:
    """Retorna dict com val_loss (nats/token) e next-token accuracy.

    Acumula loss por SOMA (reduction='sum') e divide pelo total de tokens no fim —
    isso dá a média correta mesmo com o último batch parcial, ao contrário de
    tirar a média das médias por batch.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    tot_loss, tot_correct, tot_tokens = 0.0, 0, 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        V = logits.size(-1)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="sum")
        tot_loss += loss.item()
        tot_correct += (logits.argmax(-1) == y).sum().item()
        tot_tokens += y.numel()
    return {"val_loss": tot_loss / tot_tokens, "acc": tot_correct / tot_tokens}


def bits_per_char(val_loss_nats: float, bytes_per_token: float) -> float:
    return val_loss_nats / (bytes_per_token * math.log(2))
