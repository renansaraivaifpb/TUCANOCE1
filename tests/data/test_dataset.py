"""Testes do TextDataset (nb 05) — ABLAÇÃO sliding vs chunked e contrato (x, y)."""
from __future__ import annotations

import torch

from tucanoce.data.dataset import TextDataset


def test_chunked_default_is_context_len():
    tokens = list(range(100))
    ds = TextDataset(tokens, context_len=8)
    assert ds.stride == 8                        # None => chunked


def test_sliding_vs_chunked_ratio():
    # ABLAÇÃO central (§6.2): sliding gera ~T_ctx× mais amostras que chunked
    N, T = 200, 8
    tokens = list(range(N))
    sliding = TextDataset(tokens, context_len=T, stride=1)
    chunked = TextDataset(tokens, context_len=T)
    assert len(sliding) == N - T                 # uma janela por posição
    assert len(chunked) == (N - T - 1) // T + 1
    ratio = len(sliding) / len(chunked)
    assert abs(ratio - T) < 2.0                  # razão ~ T_ctx


def test_target_is_input_shifted_by_one():
    # contrato next-token: y[t] = token[t+1]
    tokens = list(range(50))
    ds = TextDataset(tokens, context_len=8)
    x, y = ds[0]
    assert torch.equal(x[1:], y[:-1])            # y é x deslocado por 1
    assert torch.equal(y, torch.arange(1, 9))


def test_accepts_tensor_input_no_warning():
    # tokens pode vir de um cache (LongTensor); __getitem__ deve lidar sem copiar mal
    tokens = torch.arange(50, dtype=torch.long)
    ds = TextDataset(tokens, context_len=8)
    x, y = ds[0]
    assert x.dtype == torch.long and y.dtype == torch.long
    # mutar a amostra não deve afetar o buffer original (clone, não view)
    x[0] = 999
    assert tokens[0].item() == 0


def test_empty_when_too_short():
    ds = TextDataset(list(range(5)), context_len=8)
    assert len(ds) == 0
