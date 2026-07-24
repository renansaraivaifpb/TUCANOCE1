"""Testes de train (nb 06) — param groups, scheduler e convergência real."""
from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from tucanoce.config import ModelConfig, TrainConfig
from tucanoce.data.dataset import TextDataset
from tucanoce.model.transformer import TucanoCE
from tucanoce.training.train import _make_lr_lambda, build_param_groups


def test_param_groups_split_by_dim(tiny_cfg):
    model = TucanoCE(tiny_cfg)
    groups = build_param_groups(model, weight_decay=0.1)
    assert groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in groups[0]["params"])   # matrizes: com decay
    assert all(p.dim() == 1 for p in groups[1]["params"])   # 1D: sem decay


def test_tied_weight_counted_once(tiny_cfg):
    # weight tying: o peso compartilhado aparece uma vez só nos grupos
    model = TucanoCE(tiny_cfg)
    groups = build_param_groups(model, 0.1)
    all_ids = [id(p) for g in groups for p in g["params"]]
    assert len(all_ids) == len(set(all_ids))


def test_lr_schedule_warmup_then_cosine():
    total, warmup_cap, min_ratio = 100, 10, 0.1
    lr = _make_lr_lambda(total, warmup_cap, warmup_ratio=0.1, min_ratio=min_ratio)
    assert lr(0) == 0.0                          # começa em 0
    assert abs(lr(10) - 1.0) < 1e-9             # pico no fim do warmup (step=10)
    assert lr(5) < lr(10)                        # sobe durante o warmup
    assert abs(lr(total) - min_ratio) < 1e-6    # termina em min_ratio
    assert lr(50) < lr(10) and lr(50) > lr(total)   # decai monotônico no cosseno


def _learnable_loaders(vocab=16, context=8):
    # corpus cíclico: next-token = (t+1) % vocab -> perfeitamente aprendível
    cycle = list(range(vocab)) * 60
    ds_tr = TextDataset(cycle[:600], context_len=context)
    ds_va = TextDataset(cycle[600:], context_len=context)
    return (DataLoader(ds_tr, batch_size=8, shuffle=True),
            DataLoader(ds_va, batch_size=8))


def test_train_reduces_loss(tmp_path):
    from tucanoce.training.train import train
    cfg = ModelConfig(vocab_size=16, context_len=8, n_layers=2, n_heads=2,
                      embed_dim=32, dropout=0.0)
    model = TucanoCE(cfg)
    tr, va = _learnable_loaders()
    tcfg = TrainConfig(lr_max=3e-3, max_epochs=15, patience=15, warmup_cap=10,
                       use_bf16=False, batch_size=8)
    ckpt = str(tmp_path / "best.pt")
    res = train(model, tr, va, tcfg, device="cpu", ckpt_path=ckpt)
    # aprendeu a tarefa cíclica -> val_loss bem abaixo do piso log(V)
    assert res["best_val_loss"] < math.log(16) - 0.5
    assert res["best_epoch"] >= 1


def test_checkpoint_saved_and_reloadable(tmp_path):
    from tucanoce.training.train import train
    cfg = ModelConfig(vocab_size=16, context_len=8, n_layers=1, n_heads=2,
                      embed_dim=16, dropout=0.0)
    model = TucanoCE(cfg)
    tr, va = _learnable_loaders()
    tcfg = TrainConfig(lr_max=3e-3, max_epochs=3, patience=3, warmup_cap=5,
                       use_bf16=False)
    ckpt = str(tmp_path / "best.pt")
    train(model, tr, va, tcfg, device="cpu", ckpt_path=ckpt)
    # checkpoint é puro-dados: carrega com o default weights_only=True do torch>=2.6
    blob = torch.load(ckpt)
    assert "model" in blob and isinstance(blob["cfg"], dict)
    model.load_state_dict(blob["model"])
