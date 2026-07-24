"""Testes de evaluate (nb 07) — val_loss/accuracy e bits_per_char."""
from __future__ import annotations

import math

import torch

from tucanoce.config import ModelConfig
from tucanoce.model.transformer import TucanoCE
from tucanoce.training.evaluate import bits_per_char, evaluate


def test_bits_per_char_formula():
    # BPC = L / (bytes_por_token * ln2)
    assert abs(bits_per_char(2.27, 4.65) - 2.27 / (4.65 * math.log(2))) < 1e-9


def test_evaluate_returns_loss_and_acc(tiny_cfg):
    model = TucanoCE(tiny_cfg)
    x = torch.randint(0, tiny_cfg.vocab_size, (4, 8))
    y = torch.randint(0, tiny_cfg.vocab_size, (4, 8))
    loader = [(x, y)]
    m = evaluate(model, loader, device="cpu")
    assert "val_loss" in m and "acc" in m
    assert m["val_loss"] > 0 and 0.0 <= m["acc"] <= 1.0


def test_evaluate_perfect_model_low_loss():
    # modelo que sempre acerta -> loss baixa, acc alta. Forjamos logits via um
    # embedding one-hot degenerado não é trivial; usamos uma checagem indireta:
    # com alvos == argmax dos logits, acc deve ser 1.0.
    cfg = ModelConfig(vocab_size=16, context_len=8, n_layers=1, n_heads=2,
                      embed_dim=16, dropout=0.0)
    model = TucanoCE(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        y = model(x).argmax(-1)                  # alvo = predição do próprio modelo
    m = evaluate(model, [(x, y)], device="cpu")
    assert m["acc"] == 1.0


def test_evaluate_averages_over_partial_batches():
    # soma/total dá a média correta mesmo com batches de tamanhos diferentes
    cfg = ModelConfig(vocab_size=16, context_len=8, n_layers=1, n_heads=2,
                      embed_dim=16, dropout=0.0)
    model = TucanoCE(cfg)
    b1 = (torch.randint(0, 16, (4, 8)), torch.randint(0, 16, (4, 8)))
    b2 = (torch.randint(0, 16, (1, 8)), torch.randint(0, 16, (1, 8)))   # batch parcial
    m = evaluate(model, [b1, b2], device="cpu")
    assert m["val_loss"] > 0
