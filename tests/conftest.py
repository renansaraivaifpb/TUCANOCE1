"""Fixtures compartilhadas dos testes do TucanoCE.

Determinismo: cada teste que depende de init aleatório fixa a seed via a fixture
`seeded`. Modelos são minúsculos (CPU, segundos) — os testes validam PROPRIEDADES
(invariâncias, contagens, monotonicidade), não qualidade de texto.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tucanoce.config import ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def seeded():
    """Fixa seeds antes de cada teste (autouse: aplica a todos)."""
    torch.manual_seed(0)


@pytest.fixture
def tiny_cfg() -> ModelConfig:
    """Config minúscula p/ testes rápidos: dropout=0 p/ forward determinístico."""
    return ModelConfig(vocab_size=64, context_len=32, n_layers=2,
                       n_heads=2, embed_dim=16, dropout=0.0)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def toy_corpus() -> list[str]:
    return [
        "the electron is a fundamental particle with negative charge",
        "quantum mechanics describes the behavior of particles at small scales",
        "the energy of a photon is proportional to its frequency",
        "energy and mass are related by the equation e equals m c squared",
    ] * 8
