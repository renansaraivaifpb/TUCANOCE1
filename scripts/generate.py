"""Entrypoint de inferência: carrega um checkpoint treinado e gera texto.

Uso:
    python scripts/generate.py --prompt "deep learning is"
    python scripts/generate.py --config configs/ml_cpu.yaml --prompt "a neural network" \
        --max-new-tokens 80 --temperature 0.8 --top-k 40 --rep-penalty 1.2

Contrato: a arquitetura vem do PRÓPRIO checkpoint (ckpt["cfg"]), não do YAML — o
modelo salvo é a fonte da verdade da sua forma. Do YAML só sai o caminho do
tokenizer, que precisa casar com o checkpoint (tokenizer errado => texto-lixo).

É um LM BASE (só pré-treino): ele CONTINUA texto, não segue instruções.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tucanoce.config import ModelConfig, load_configs
from tucanoce.inference.generate import generate
from tucanoce.model.transformer import TucanoCE
from tucanoce.tokenizer.bpe import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ml_cpu.yaml",
                    help="de onde ler o tokenizer_path (deve casar com o checkpoint)")
    ap.add_argument("--ckpt", default=None, help="override do checkpoint")
    ap.add_argument("--tokenizer", default=None, help="override do tokenizer")
    ap.add_argument("--prompt", default="a neural network")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--rep-penalty", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    _, _, data_cfg = load_configs(args.config)
    ckpt_path = args.ckpt or "checkpoints/best.pt"
    tok_path = args.tokenizer or data_cfg.get("tokenizer_path", "data/tokenizer.json")

    if not os.path.exists(ckpt_path):
        raise SystemExit(f"checkpoint não encontrado: {ckpt_path} — treine antes com "
                         f"`python scripts/pretrain.py --config {args.config}`")

    # arquitetura vem do checkpoint; tokenizer vem do disco
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model = TucanoCE(ModelConfig(**ckpt["cfg"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer = BPETokenizer.load(tok_path)

    vl = ckpt.get("val_loss")
    print(f"modelo: {model.num_params():,} params · vocab {model.cfg.vocab_size} · "
          f"ctx {model.cfg.context_len} · val_loss "
          f"{vl:.3f}" if vl is not None else "?", f"· tokenizer {tok_path}")

    gen = None if args.seed < 0 else torch.Generator().manual_seed(args.seed)
    text = generate(
        model, tokenizer, prompt=args.prompt,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if args.top_p > 0 else None,
        repetition_penalty=args.rep_penalty, device=args.device, generator=gen,
    )
    print(f"\n--- geração (prompt: {args.prompt!r}) ---\n{text}\n")


if __name__ == "__main__":
    main()
