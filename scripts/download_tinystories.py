"""Baixa o TinyStories (Eldan & Li, 2023) e converte para o JSONL do projeto.

TinyStories é o corpus certo para o REGIME deste projeto: historinhas sintéticas
com vocabulário de criança de 3-4 anos. Modelos de 1-33M params — a nossa faixa —
produzem texto FLUENTE nele, ao contrário de prosa técnica (física/ML) de alta
entropia. Ver arXiv:2305.07759.

Uso:
    python scripts/download_tinystories.py                       # ~8000 histórias
    python scripts/download_tinystories.py --max-stories 20000   # corpus maior
    python scripts/download_tinystories.py --split train         # o arquivo cheio (~2GB)

Depois: python scripts/pretrain.py --config configs/tinystories_cpu.yaml

Baixa direto do HuggingFace (sem a lib `datasets`): os arquivos são texto plano com
histórias separadas por '<|endoftext|>'. O `valid` (~19 MB) já é subconjunto limpo e
mais que suficiente para CPU; `train` é o corpus completo (grande demais p/ CPU).
"""
from __future__ import annotations

import argparse
import json
import os

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
EOT_SEP = "<|endoftext|>"


def download_text(split: str) -> str:
    """Baixa TinyStories-{split}.txt em streaming e devolve o texto inteiro."""
    import requests

    url = f"{BASE}/TinyStories-{split}.txt"
    print(f"baixando {url} ...")
    chunks, got = [], 0
    with requests.get(url, stream=True, timeout=60,
                      headers={"User-Agent": "tucanoce/0.1 (educational)"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        for chunk in r.iter_content(chunk_size=1 << 20):   # 1 MB
            chunks.append(chunk)
            got += len(chunk)
            if total:
                print(f"\r  {got/1e6:5.1f}/{total/1e6:.1f} MB", end="", flush=True)
    print()
    return b"".join(chunks).decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid", choices=["valid", "train"])
    ap.add_argument("--max-stories", type=int, default=8000,
                    help="teto de histórias (controla o tempo de treino em CPU)")
    ap.add_argument("--min-chars", type=int, default=200, help="descarta histórias curtas")
    ap.add_argument("--out", default="data/corpus_tinystories.jsonl")
    args = ap.parse_args()

    raw = download_text(args.split)
    # histórias separadas por <|endoftext|>; limpeza mínima (o corpus já é limpo)
    stories = (s.strip() for s in raw.split(EOT_SEP))

    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

    n, chars = 0, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for s in stories:
            if len(s) < args.min_chars:
                continue
            f.write(json.dumps({
                "text": s, "title": f"story_{n}",
                "source": "tinystories", "category": args.split,
            }, ensure_ascii=False) + "\n")
            n += 1
            chars += len(s)
            if n >= args.max_stories:
                break

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"concluído: {n} histórias, {chars/1e6:.2f}M chars (~{chars//4//1000}K tokens "
          f"aprox), {size_mb:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
