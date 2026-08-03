"""Caracteriza quantitativamente a entropia dos corpora.

O estudo de corpora afirma que o TinyStories é "de entropia mais baixa" que prosa
técnica. Sem medida, isso é qualitativo. Este script calcula quatro proxies
intrínsecos (independentes do tamanho do corpus, exceto onde indicado):

1. bytes/token — compressão que o BPE consegue; texto mais previsível comprime mais.
2. tipos usados — quantos ids do vocabulário aparecem de fato.
3. H1, entropia de unigrama (nats/token) — dispersão da distribuição marginal.
4. H2, entropia condicional de bigrama H(x_t | x_{t-1}) (nats/token) — quanto de
   incerteza sobra conhecendo só o token anterior.

H2 é a referência mais útil: é a cross-entropy que um contador de bigramas
atingiria, ou seja, o piso que o modelo precisa BATER para ter aprendido algo além
de coocorrência local. Ressalva: é estimador plug-in in-sample, portanto otimista
(subestima o valor held-out de um bigrama real). A comparação entre corpora é
válida porque os três são calculados de forma idêntica.

Uso:
    python scripts/entropia_corpora.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter

import torch

# (nome, cache de tokens, jsonl do texto cru)
CORPORA = [
    ("Física de partículas", "data/corpus_tokens_physics.pt", "data/corpus_physics.jsonl"),
    ("Machine learning", "data/corpus_tokens_ml.pt", "data/corpus.jsonl"),
    ("TinyStories", "data/corpus_tokens_tinystories.pt", "data/corpus_tinystories.jsonl"),
]

# val_loss medido do preset small (1,8M) em cada corpus, para a coluna de margem
VAL_LOSS = {
    "Física de partículas": 3.087,
    "Machine learning": 3.017,
    "TinyStories": 1.586,
}


def entropias(ids: list[int]) -> tuple[float, float, int]:
    """Devolve (H1, H2, n_tipos) em nats/token.

    H1 = -sum p(x) log p(x).
    H2 = -sum p(a,b) log p(b|a), com p(b|a) = c(a,b)/c(a).
    Custo: O(N) tempo e O(|bigramas distintos|) memória.
    """
    n = len(ids)
    c1 = Counter(ids)
    h1 = -sum((v / n) * math.log(v / n) for v in c1.values())

    cb = Counter(zip(ids, ids[1:]))
    m = n - 1
    h2 = -sum((v / m) * math.log(v / c1[a]) for (a, _b), v in cb.items())
    return h1, h2, len(c1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/entropia_corpora.json")
    ap.add_argument("--vocab", type=int, default=4096, help="V, para o piso trivial log V")
    args = ap.parse_args()

    piso = math.log(args.vocab)
    rows = []

    hdr = (f"{'corpus':22} {'tokens':>10} {'B/tok':>6} {'tipos':>6} "
           f"{'H1':>6} {'H2':>6} {'val_loss':>9} {'margem':>7}")
    print(hdr)
    print("-" * len(hdr))

    for name, tok_path, raw_path in CORPORA:
        if not (os.path.exists(tok_path) and os.path.exists(raw_path)):
            print(f"{name:22} (arquivos ausentes — pulando)")
            continue
        ids = torch.load(tok_path)["tokens"].tolist()
        nbytes = sum(len(json.loads(l)["text"].encode("utf-8"))
                     for l in open(raw_path, encoding="utf-8"))
        h1, h2, tipos = entropias(ids)
        vl = VAL_LOSS.get(name)
        margem = (h2 - vl) if vl is not None else None

        rows.append(dict(corpus=name, tokens=len(ids), bytes=nbytes,
                         bytes_por_token=nbytes / len(ids), tipos=tipos,
                         H1=h1, H2=h2, val_loss=vl, margem_sobre_bigrama=margem))
        print(f"{name:22} {len(ids):>10,} {nbytes/len(ids):>6.2f} {tipos:>6} "
              f"{h1:>6.3f} {h2:>6.3f} {vl:>9.3f} {margem:>7.3f}")

    print(f"\npiso trivial log({args.vocab}) = {piso:.3f} nats")
    print("margem = H2 - val_loss: quanto o modelo ganha sobre um contador de bigramas")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"piso_trivial": piso, "vocab": args.vocab, "corpora": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
