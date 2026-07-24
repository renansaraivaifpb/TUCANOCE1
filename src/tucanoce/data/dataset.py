"""TextDataset com stride configurável (paper seção 6.2, Listing 5).

O gargalo escondido: sliding window de stride=1 gera N-T_ctx amostras (cada token
em T_ctx amostras). Chunked (stride=T_ctx) gera ~N/T_ctx amostras, cada token em
exatamente uma amostra por epoch. Ganho ~124x em steps/epoch sem perda de qualidade
(seção 6.2.4). Default = chunked; stride=1 fica para ablações.

Ver notebook 05_pipeline_dados.ipynb.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, tokens, context_len: int, stride: int | None = None):
        self.tokens = tokens
        self.context_len = context_len
        self.stride = stride if stride is not None else context_len

    def __len__(self):
        if len(self.tokens) < self.context_len + 1:
            return 0
        return (len(self.tokens) - self.context_len - 1) // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.tokens[start: start + self.context_len]
        y = self.tokens[start + 1: start + self.context_len + 1]
        # tokens pode ser uma lista OU um LongTensor pré-carregado (cache): fatiar
        # um tensor já devolve tensor — clonamos p/ evitar o UserWarning e não
        # compartilhar storage entre amostras; de lista, construímos o tensor.
        to_long = lambda s: (s.detach().clone().long() if torch.is_tensor(s)
                             else torch.tensor(s, dtype=torch.long))
        return to_long(x), to_long(y)
