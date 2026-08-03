"""Loop de treino (nb 06).

Componentes:
- Objetivo: cross-entropy autoregressiva, F.cross_entropy(logits, targets).
- AdamW com weight decay separado: decay=0.1 em matrizes Linear, decay=0 em
  biases/normalizações (parâmetros 1D) — nb 06.
- Scheduler cosine com warmup linear (LOSHCHILOV; HUTTER, 2017); T_w = min(T/10, 2000).
- BF16 via torch.amp.autocast; dispensa GradScaler (nb 06).
- Gradient clipping por norma global em 1.0 (PASCANU et al., 2013) + gradient accumulation.
- Early stopping com save-best por val_loss (nb 06).

Ver notebook 06_treinamento.ipynb.
"""
from __future__ import annotations

import dataclasses
import math
import os

import torch
import torch.nn.functional as F

from .evaluate import evaluate


def build_param_groups(model, weight_decay: float):
    """Separa parâmetros 2D+ (com decay) de 1D (sem decay). nb 06.

    Decaimento faz sentido em matrizes (regularização), mas penalizar o gamma de
    uma norma ou um bias — que controlam ESCALA/deslocamento — distorce o
    equilíbrio em vez de regularizar. Regra: decaia 2D+, deixe 1D livres. O peso
    compartilhado (weight tying) aparece uma vez só em named_parameters().
    """
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _make_lr_lambda(total_steps: int, warmup_cap: int, warmup_ratio: float,
                    min_ratio: float):
    """Multiplicador de lr_max: warmup linear + decaimento cosseno até min_ratio (LOSHCHILOV; HUTTER, 2017)."""
    warmup = min(int(total_steps * warmup_ratio), warmup_cap)
    warmup = max(warmup, 1)

    def lr_lambda(step: int) -> float:
        if step < warmup:                               # sobe linear: evita passos
            return step / warmup                        # destrutivos no início ruidoso
        prog = (step - warmup) / max(total_steps - warmup, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * prog))   # desce suave até lr_min
        return min_ratio + (1 - min_ratio) * cosine

    return lr_lambda


def train(model, train_loader, val_loader, cfg, device=None,
          ckpt_path: str = "checkpoints/best.pt"):
    """Loop de pré-treino completo (nb 06). Retorna dict com melhor val_loss, epoch e histórico.

    Fia as seis peças: cross-entropy autoregressiva, AdamW com weight decay
    separado, scheduler cosseno+warmup, autocast BF16 (sem GradScaler), grad
    clipping + accumulation, e early stopping com save-best.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    # BF16 só quando há GPU compatível; na CPU roda FP32 (mesma estrutura de código).
    use_bf16 = cfg.use_bf16 and device_type == "cuda" and torch.cuda.is_bf16_supported()

    optimizer = torch.optim.AdamW(
        build_param_groups(model, cfg.weight_decay),
        lr=cfg.lr_max, betas=cfg.betas, eps=cfg.eps,
    )
    # total_steps conta optimizer.step() (1 a cada grad_accum micro-batches).
    steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.grad_accum))
    total_steps = cfg.max_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _make_lr_lambda(total_steps, cfg.warmup_cap,
                                   cfg.warmup_ratio, cfg.lr_min_ratio))

    best_val, no_improve, best_epoch = float("inf"), 0, -1
    history: list[dict] = []
    if os.path.dirname(ckpt_path):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    n_batches = len(train_loader)

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                                enabled=use_bf16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                ) / cfg.grad_accum                       # escala p/ somar gradientes
            loss.backward()
            # optimizer.step() a cada grad_accum micro-batches (e no último batch)
            if (i + 1) % cfg.grad_accum == 0 or (i + 1) == n_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        metrics = evaluate(model, val_loader, device=device)
        val_loss = metrics["val_loss"]
        history.append({"epoch": epoch, "val_loss": val_loss, "acc": metrics["acc"]})

        if val_loss < best_val - 1e-4:                   # save-best: salva o MELHOR,
            best_val, best_epoch, no_improve = val_loss, epoch, 0
            # cfg como dict (não o objeto): mantém o checkpoint puro-dados, seguro
            # p/ torch.load(weights_only=True), o default do PyTorch >= 2.6.
            torch.save({"model": model.state_dict(),
                        "cfg": dataclasses.asdict(model.cfg),
                        "val_loss": val_loss, "epoch": epoch}, ckpt_path)
            tag = "  <- best (checkpoint salvo)"
        else:
            no_improve += 1
            tag = f"  (no_improve={no_improve})"

        print(f"epoch {epoch:2d} | val_loss {val_loss:.4f} | acc {metrics['acc']:.3f} "
              f"| lr {scheduler.get_last_lr()[0]:.2e}{tag}")

        if no_improve >= cfg.patience:                   # patience stop
            print(f">> Early stopping no epoch {epoch}: {cfg.patience} epochs sem melhora.")
            break

    return {"best_val_loss": best_val, "best_epoch": best_epoch,
            "history": history, "ckpt_path": ckpt_path}
