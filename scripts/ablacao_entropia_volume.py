"""Ablação que separa entropia do corpus de volume de dados.

No estudo de corpora (física -> ML -> TinyStories) os dois fatores estão
confundidos: o TinyStories não é só de entropia mais baixa, ele também tem ~7x
mais tokens. Sem desacoplar, não se pode atribuir a queda de val_loss à
entropia.

O controle: truncar o TinyStories ao MESMO número de tokens do corpus de física
(415.878) e treinar com os hiperparâmetros do run de física — preset small,
context_len 128, chunked, max_epochs 12, patience 4, batch 16, val_frac 0,1.
Assim o corpus é o único fator que varia.

Uso:
    python scripts/ablacao_entropia_volume.py
    python scripts/ablacao_entropia_volume.py --n-tokens 415878 --out results/

Grava métricas por época em JSON (a curva do artigo é reproduzível a partir daí)
e o checkpoint num caminho próprio — nunca sobrescreve checkpoints/best.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tucanoce.config import load_configs
from tucanoce.data.dataset import TextDataset
from tucanoce.inference.generate import generate
from tucanoce.model.transformer import TucanoCE
from tucanoce.tokenizer.bpe import BPETokenizer
from tucanoce.training.train import train

# volume do corpus de física de partículas: o alvo a igualar
N_TOKENS_FISICA = 415_878


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/physics_cpu.yaml",
                    help="hiperparâmetros do run de referência (física)")
    ap.add_argument("--tokenizer", default="data/tokenizer_tinystories.json")
    ap.add_argument("--tokens-cache", default="data/corpus_tokens_tinystories.pt")
    ap.add_argument("--n-tokens", type=int, default=N_TOKENS_FISICA)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default="results")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1337,
                    help="semente: sem ela o número reportado não é reproduzível")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # init dos pesos e shuffle do DataLoader consomem o RNG global do torch
    torch.manual_seed(args.seed)

    # hiperparâmetros do run de física: isola o corpus como único fator
    model_cfg, train_cfg, _ = load_configs(args.config)

    tokenizer = BPETokenizer.load(args.tokenizer)
    tokens_full = torch.load(args.tokens_cache)["tokens"]
    if tokens_full.numel() < args.n_tokens:
        raise SystemExit(f"cache tem {tokens_full.numel()} tokens < alvo {args.n_tokens}")
    tokens = tokens_full[:args.n_tokens]
    print(f"tokens: {tokens.numel():,} (truncado de {tokens_full.numel():,})")

    model_cfg.vocab_size = tokenizer.actual_vocab_size

    split = int((1 - args.val_frac) * len(tokens))
    train_ds = TextDataset(tokens[:split], model_cfg.context_len, None)  # chunked
    val_ds = TextDataset(tokens[split:], model_cfg.context_len, None)
    train_ld = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=train_cfg.batch_size)
    print(f"dataset: train {len(train_ds)} | val {len(val_ds)} amostras")

    model = TucanoCE(model_cfg)
    print(f"modelo: {model.num_params():,} parâmetros")

    ckpt_path = os.path.join(args.out, "ablacao_entropia_volume.pt")
    result = train(model, train_ld, val_ld, train_cfg, device=args.device,
                   ckpt_path=ckpt_path)

    model.load_state_dict(torch.load(ckpt_path)["model"])
    sample = generate(model, tokenizer, prompt=args.prompt, max_new_tokens=60,
                     temperature=0.8, top_k=40, repetition_penalty=1.2)

    # a curva por época é o insumo da figura do artigo: persistir é o que a torna
    # reproduzível (o dict `history` do train() morria em memória)
    payload = {
        "n_tokens": int(tokens.numel()),
        "n_params": model.num_params(),
        "config": args.config,
        "best_val_loss": result["best_val_loss"],
        "best_epoch": result["best_epoch"],
        "history": result["history"],
        "sample": sample,
    }
    out_json = os.path.join(args.out, "ablacao_entropia_volume.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nbest_val_loss {result['best_val_loss']:.4f} (epoch {result['best_epoch']})")
    print(f"métricas -> {out_json}")
    print(f"\n--- amostra (prompt: {args.prompt!r}) ---\n{sample}")


if __name__ == "__main__":
    main()
