"""Testes de config.py — presets, hidden_dim do SwiGLU e loader YAML."""
from __future__ import annotations

import pytest

from tucanoce.config import (ModelConfig, TrainConfig, compute_hidden_dim,
                           get_model_config, load_configs)


@pytest.mark.parametrize("d,expected", [(512, 1408), (128, 384), (256, 704), (768, 2048)])
def test_hidden_dim_multiple_of_64(d, expected):
    h = compute_hidden_dim(d)
    assert h == expected
    assert h % 64 == 0                      # alinhamento tensor core
    assert h >= 8 * d / 3                   # arredonda p/ cima


def test_head_dim_divides():
    cfg = get_model_config("medium")
    assert cfg.embed_dim % cfg.n_heads == 0
    assert cfg.head_dim == cfg.embed_dim // cfg.n_heads


def test_post_init_computes_hidden():
    cfg = ModelConfig(embed_dim=512, hidden_dim=None)
    assert cfg.hidden_dim == compute_hidden_dim(512)


def test_invalid_heads_raises():
    with pytest.raises(AssertionError):
        ModelConfig(embed_dim=10, n_heads=3)   # 10 % 3 != 0


def test_presets_exist():
    for name in ["small", "base", "medium", "large", "xl"]:
        cfg = get_model_config(name)
        assert isinstance(cfg, ModelConfig)


def test_overrides():
    cfg = get_model_config("medium", vocab_size=1000, dropout=0.0)
    assert cfg.vocab_size == 1000 and cfg.dropout == 0.0
    assert cfg.embed_dim == 512             # preset preservado


def test_load_configs_yaml(repo_root):
    mc, tc, data = load_configs(str(repo_root / "configs" / "small.yaml"))
    assert isinstance(mc, ModelConfig) and isinstance(tc, TrainConfig)
    assert mc.embed_dim == 128 and mc.context_len == 64
    # betas vem como lista no YAML -> deve virar tupla (o que o AdamW espera)
    assert isinstance(tc.betas, tuple)
    assert "corpus_path" in data
