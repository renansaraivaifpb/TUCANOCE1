"""Benchmark honesto: TucanoCE (nosso, treinado no domínio) vs GPT-2 (124M, zero-shot).

Por que BITS POR BYTE (BPB) e não perplexity: perplexity é medida por TOKEN, e os
dois modelos usam tokenizers diferentes (nosso vocab 4096 vs 50257 do GPT-2) — são
unidades incomparáveis. BPB normaliza a log-verossimilhança pelos BYTES do texto
cru, que independem do tokenizer. É a única comparação apples-to-apples.

O que o benchmark mede, sobre um conjunto de HISTÓRIAS QUE NENHUM DOS DOIS TREINOU
(held-out do TinyStories):
  1. BPB (menor = melhor) — quão bem cada modelo comprime/prevê o texto.
  2. Amostras de geração lado a lado (mesmo prompt).
  3. Contexto: params, dados de treino, velocidade de inferência em CPU.

Nuance justa: GPT-2 é generalista e NUNCA viu TinyStories (zero-shot); o nosso é um
especialista treinado no domínio. O benchmark responde: um especialista 70x menor
compete com um generalista gigante, no domínio dele?

Uso:
    python scripts/benchmark.py                       # ~150 histórias held-out
    python scripts/benchmark.py --n-eval 300 --gpt2 gpt2-medium
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tucanoce.config import ModelConfig, load_configs
from tucanoce.inference.generate import generate
from tucanoce.model.transformer import TucanoCE
from tucanoce.tokenizer.bpe import BPETokenizer

TINY_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
LN2 = math.log(2)


def load_heldout(skip: int, n: int, min_chars: int = 200) -> str:
    """Baixa o valid do TinyStories e devolve n histórias APÓS as `skip` primeiras
    (as `skip` primeiras foram o nosso treino — held-out real)."""
    import requests
    print(f"baixando held-out do TinyStories (pulando {skip}, pegando {n})...")
    r = requests.get(TINY_URL, timeout=120, headers={"User-Agent": "tucanoce/0.1"})
    r.raise_for_status()
    stories = [s.strip() for s in r.text.split("<|endoftext|>") if len(s.strip()) >= min_chars]
    chosen = stories[skip:skip + n]
    if not chosen:
        raise SystemExit(f"held-out vazio (só há {len(stories)} histórias; reduza --skip)")
    return "\n".join(chosen)


@torch.no_grad()
def stream_nll_nats(forward_logits, ids: list[int], ctx: int, device) -> tuple[float, int]:
    """NLL total (nats) do stream `ids` por janela deslizante (recipe HF), com
    stride = ctx//2 e mascarando o overlap p/ cada token ser pontuado UMA vez com
    o máximo de contexto. `forward_logits(input_1xT) -> logits[1,T,V]`.
    Retorna (soma_nll_nats, n_tokens_pontuados)."""
    stride = max(1, ctx // 2)
    seq = len(ids)
    total_nll, n_scored, prev_end = 0.0, 0, 0
    for begin in range(0, seq, stride):
        end = min(begin + ctx, seq)
        trg_len = end - prev_end                       # tokens novos a pontuar
        inp = torch.tensor([ids[begin:end]], dtype=torch.long, device=device)
        logits = forward_logits(inp)[0]                # [L, V]
        shift_logits = logits[:-1]                     # prevê o próximo
        shift_tgt = inp[0, 1:].clone()                 # [L-1]
        ignore = (end - begin - 1) - trg_len           # ignora o que já foi pontuado
        if ignore > 0:
            shift_tgt[:ignore] = -100
        if shift_logits.size(0) > 0:
            loss = F.cross_entropy(shift_logits, shift_tgt, ignore_index=-100,
                                   reduction="sum")
            total_nll += loss.item()
            n_scored += int((shift_tgt != -100).sum())
        prev_end = end
        if end == seq:
            break
    return total_nll, n_scored


def bpb(nll_nats: float, total_bytes: int) -> float:
    return nll_nats / (total_bytes * LN2)


# ----------------------------------------------------------- nosso modelo
def load_ours(config: str, ckpt_path: str):
    _, _, data_cfg = load_configs(config)
    tok = BPETokenizer.load(data_cfg["tokenizer_path"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = TucanoCE(ModelConfig(**ckpt["cfg"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok, ckpt.get("val_loss")


# ----------------------------------------------------------- métricas
def eval_ours(model, tok, text: str, total_bytes: int) -> dict:
    ids = tok.encode(text)
    nll, n = stream_nll_nats(lambda x: model(x), ids, model.cfg.context_len, "cpu")
    return {"bpb": bpb(nll, total_bytes), "ppl": math.exp(nll / n),
            "n_tokens": n, "ctx": model.cfg.context_len}


def eval_gpt2(name: str, text: str, total_bytes: int) -> dict:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    print(f"carregando {name} (pode baixar ~500MB na 1ª vez)...")
    tok = GPT2TokenizerFast.from_pretrained(name)
    model = GPT2LMHeadModel.from_pretrained(name).eval()
    ids = tok(text)["input_ids"]
    ctx = model.config.n_positions                     # 1024
    nll, n = stream_nll_nats(lambda x: model(x).logits, ids, ctx, "cpu")
    n_params = sum(p.numel() for p in model.parameters())
    return {"bpb": bpb(nll, total_bytes), "ppl": math.exp(nll / n),
            "n_tokens": n, "ctx": ctx, "params": n_params,
            "_model": model, "_tok": tok}


def sample_gpt2(bundle: dict, prompt: str, max_new: int) -> tuple[str, float]:
    model, tok = bundle["_model"], bundle["_tok"]
    ids = tok(prompt, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                             temperature=0.8, top_k=40, pad_token_id=tok.eos_token_id)
    dt = time.time() - t0
    txt = tok.decode(out[0], skip_special_tokens=True)
    return txt, max_new / dt


def sample_ours(model, tok, prompt: str, max_new: int) -> tuple[str, float]:
    t0 = time.time()
    txt = generate(model, tok, prompt=prompt, max_new_tokens=max_new,
                   temperature=0.8, top_k=40, repetition_penalty=1.2, device="cpu")
    dt = time.time() - t0
    return txt, max_new / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tinystories_cpu.yaml")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--gpt2", default="gpt2", help="gpt2 | gpt2-medium | ...")
    ap.add_argument("--n-eval", type=int, default=150, help="histórias held-out")
    ap.add_argument("--skip", type=int, default=8000, help="pula as treinadas")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new", type=int, default=80)
    args = ap.parse_args()

    text = load_heldout(args.skip, args.n_eval)
    total_bytes = len(text.encode("utf-8"))
    print(f"held-out: {args.n_eval} histórias, {total_bytes/1e3:.1f} KB "
          f"({total_bytes} bytes)\n")

    print("== avaliando NOSSO modelo (TucanoCE) ==")
    ours, otok, oval = load_ours(args.config, args.ckpt)
    o = eval_ours(ours, otok, text, total_bytes)
    o["params"] = ours.num_params()

    print(f"== avaliando {args.gpt2} (zero-shot) ==")
    g = eval_gpt2(args.gpt2, text, total_bytes)

    otxt, ospd = sample_ours(ours, otok, args.prompt, args.max_new)
    gtxt, gspd = sample_gpt2(g, args.prompt, args.max_new)

    # ------------------------------------------------ relatório
    print("\n" + "=" * 68)
    print("BENCHMARK — TucanoCE (especialista, in-domain) vs GPT-2 (zero-shot)")
    print("=" * 68)
    rows = [
        ("parâmetros", f"{o['params']:,}", f"{g['params']:,}"),
        ("treinou em TinyStories?", "sim (1.66M tok)", "não (zero-shot)"),
        ("contexto (tokens)", o["ctx"], g["ctx"]),
        ("BPB  (bits/byte) ↓", f"{o['bpb']:.3f}", f"{g['bpb']:.3f}"),
        ("perplexity (por token)*", f"{o['ppl']:.1f}", f"{g['ppl']:.1f}"),
        ("geração (tok/s CPU) ↑", f"{ospd:.1f}", f"{gspd:.1f}"),
    ]
    w = max(len(r[0]) for r in rows)
    print(f"\n{'métrica':<{w}} | {'TucanoCE':>16} | {'GPT-2':>16}")
    print("-" * (w + 38))
    for name, a, b in rows:
        print(f"{name:<{w}} | {str(a):>16} | {str(b):>16}")
    print("\n* perplexity NÃO é comparável entre tokenizers (vocabs diferentes); "
          "está aqui só como referência interna. A métrica JUSTA é o BPB.")

    winner = "TucanoCE" if o["bpb"] < g["bpb"] else args.gpt2
    print(f"\n>> Menor BPB (melhor modelagem do held-out): {winner}")

    print(f"\n--- amostra: TucanoCE (prompt {args.prompt!r}) ---\n{otxt}")
    print(f"\n--- amostra: {args.gpt2} (prompt {args.prompt!r}) ---\n{gtxt}\n")


if __name__ == "__main__":
    main()
