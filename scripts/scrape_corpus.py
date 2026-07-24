"""Coleta o corpus real: artigos de machine learning da Wikipedia em inglês -> JSONL limpo.

Uso:
    python scripts/scrape_corpus.py --per-category 200 --out data/corpus.jsonl

Depois: python scripts/pretrain.py --config configs/ml_cpu.yaml  (usa esse JSONL).

Nota de curadoria: como física, artigos de ML são densos em fórmulas — o
`_remove_math_markup` de scrape.py continua essencial aqui (senão o modelo aprende
a cuspir LaTeX). Ver notebook 05.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tucanoce.data.scrape import scrape_categories

# Categorias de machine learning densas em inglês (§6.1): profundidade técnica alta.
ML_CATEGORIES = [
    "Machine_learning", "Artificial_neural_networks", "Deep_learning",
    "Artificial_intelligence", "Natural_language_processing", "Computer_vision",
    "Reinforcement_learning", "Supervised_learning", "Unsupervised_learning",
    "Statistical_classification", "Cluster_analysis", "Data_mining",
    "Pattern_recognition", "Regression_analysis", "Dimensionality_reduction",
    "Ensemble_learning", "Bayesian_statistics",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=200)
    ap.add_argument("--out", default="data/corpus.jsonl")
    ap.add_argument("--no-subcats", action="store_true",
                    help="não descer em subcategorias (só artigos diretos)")
    args = ap.parse_args()

    print(f"scraping {len(ML_CATEGORIES)} categorias de ML "
          f"(até {args.per_category} títulos/cat) -> {args.out}\n")
    n = scrape_categories(ML_CATEGORIES, per_category=args.per_category,
                          out_path=args.out, include_subcats=not args.no_subcats)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nconcluído: {n} artigos, {size_mb:.1f} MB em {args.out}")


if __name__ == "__main__":
    main()
