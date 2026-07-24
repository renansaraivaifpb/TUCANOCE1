"""Testes do TucanoCE (nb 03/04) — forward, weight tying, init 1/√(2L), num_params, cache."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tucanoce.config import get_model_config
from tucanoce.model.transformer import TucanoCE


def test_forward_shape(tiny_cfg):
    model = TucanoCE(tiny_cfg).eval()
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 8))
    logits = model(x)
    assert logits.shape == (2, 8, tiny_cfg.vocab_size)


def test_weight_tying_same_storage(tiny_cfg):
    model = TucanoCE(tiny_cfg)
    assert model.embed.weight is model.lm_head.weight
    assert model.embed.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_residual_init_scaled(tiny_cfg):
    # projeções residuais (out_proj, down) ~ 0.02/sqrt(2L); Linear comum ~0.02
    model = TucanoCE(tiny_cfg)
    esperado = 0.02 / math.sqrt(2 * tiny_cfg.n_layers)
    std_res = model.blocks[0].attn.out_proj.weight.std().item()
    std_q = model.blocks[0].attn.q_proj.weight.std().item()
    assert std_res < std_q
    assert abs(std_res - esperado) < 0.01


def test_init_loss_near_log_vocab(tiny_cfg):
    # sanidade do init: no passo 0 a loss deve rondar log(V) (piso trivial)
    model = TucanoCE(tiny_cfg).eval()
    x = torch.randint(0, tiny_cfg.vocab_size, (4, 16))
    y = torch.randint(0, tiny_cfg.vocab_size, (4, 16))
    loss = F.cross_entropy(model(x).reshape(-1, tiny_cfg.vocab_size), y.reshape(-1))
    assert abs(loss.item() - math.log(tiny_cfg.vocab_size)) < 0.5


def test_num_params_matches_paper():
    # weight tying deduplicado por parameters(): medium ~= 42.7M (paper ~43M)
    assert abs(TucanoCE(get_model_config("medium")).num_params() - 42_742_272) < 1000


def test_forward_with_cache_shapes(tiny_cfg):
    model = TucanoCE(tiny_cfg).eval()
    x = torch.randint(0, tiny_cfg.vocab_size, (1, 5))
    logits, kvs = model(x, use_cache=True)
    assert logits.shape == (1, 5, tiny_cfg.vocab_size)
    assert len(kvs) == tiny_cfg.n_layers        # um (k,v) por bloco


def test_generate_cache_equals_recompute(tiny_cfg):
    # forward incremental com cache == recompute do prefixo inteiro
    model = TucanoCE(tiny_cfg).eval()
    seq = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
    with torch.no_grad():
        full = torch.stack([model(seq[:, :t])[:, -1] for t in range(1, 9)], dim=1)
        cached, past = [], None
        for t in range(8):
            logits, past = model(seq[:, t:t+1], past_kvs=past, use_cache=True)
            cached.append(logits[:, -1])
        cached = torch.stack(cached, dim=1)
    assert (full - cached).abs().max() < 1e-4
