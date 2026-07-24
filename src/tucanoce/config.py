"""Configuração central do modelo e do treino.

Decisão de arquitetura: uma única fonte da verdade para hiperparâmetros, em vez
de defaults espalhados pelo código (o erro nº1 apontado na seção 7 do blueprint).
Presets vêm da Tabela de escalonamento do paper (seção 5.2.4 / 8.5).

`compute_hidden_dim` reproduz a Eq. (36)-(37): SwiGLU usa hidden ~ 8d/3 (para
casar a contagem de parâmetros da MLP GELU-4d do GPT-2), arredondado para múltiplo
de 64 (alinhamento com o tile dos tensor cores da NVIDIA).
"""
from __future__ import annotations

from dataclasses import dataclass, field


def compute_hidden_dim(embed_dim: int, multiple_of: int = 64) -> int:
    hidden = int(8 * embed_dim / 3)
    return ((hidden + multiple_of - 1) // multiple_of) * multiple_of


@dataclass
class ModelConfig:
    vocab_size: int = 8192       # seção 3.3 — pequeno de propósito p/ corpus pequeno
    context_len: int = 512       # T_ctx
    n_layers: int = 12           # L
    n_heads: int = 8             # H
    embed_dim: int = 512         # d
    rope_base: float = 10000.0   # seção 5.3.5
    dropout: float = 0.1
    # hidden da MLP SwiGLU; None => calculado por compute_hidden_dim
    hidden_dim: int | None = None

    def __post_init__(self):
        if self.hidden_dim is None:
            self.hidden_dim = compute_hidden_dim(self.embed_dim)
        assert self.embed_dim % self.n_heads == 0, "embed_dim deve dividir n_heads"

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.n_heads


# Presets da Tabela de escalonamento (d, L, H) — seção 5.2.4 / 8.5.
PRESETS: dict[str, dict] = {
    "small":  dict(embed_dim=128, n_layers=6,  n_heads=4),
    "base":   dict(embed_dim=256, n_layers=8,  n_heads=8),
    "medium": dict(embed_dim=512, n_layers=12, n_heads=8),   # 43M — o "carro-chefe"
    "large":  dict(embed_dim=768, n_layers=12, n_heads=12),  # 91M
    "xl":     dict(embed_dim=1024, n_layers=16, n_heads=16), # 211M
}


def get_model_config(preset: str = "medium", **overrides) -> ModelConfig:
    cfg = dict(PRESETS[preset])
    cfg.update(overrides)
    return ModelConfig(**cfg)


def load_configs(path: str) -> tuple[ModelConfig, "TrainConfig", dict]:
    """Lê um YAML (ex.: configs/medium.yaml) e devolve (ModelConfig, TrainConfig, data).

    Mantém a config como única fonte da verdade: o script de treino não hardcoda
    hiperparâmetro nenhum, só consome o que sai daqui. `betas` no YAML vem como
    lista — convertemos p/ tupla (o que o AdamW espera).
    """
    import yaml

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    m = dict(raw.get("model", {}))
    preset = m.pop("preset", "medium")
    model_cfg = get_model_config(preset, **m)

    t = dict(raw.get("train", {}))
    if "betas" in t:
        t["betas"] = tuple(t["betas"])
    train_cfg = TrainConfig(**t)

    return model_cfg, train_cfg, dict(raw.get("data", {}))


@dataclass
class TrainConfig:
    lr_max: float = 3e-4          # seção 7.2
    lr_min_ratio: float = 0.1     # lr_min = 0.1 * lr_max
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    warmup_ratio: float = 0.1     # T_w = min(T/10, 2000)
    warmup_cap: int = 2000
    grad_clip: float = 1.0        # norma global
    grad_accum: int = 1
    max_epochs: int = 20
    patience: int = 4             # early stopping
    batch_size: int = 16
    use_bf16: bool = True
