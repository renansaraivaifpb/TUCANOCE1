"""Entrypoint de pré-treino: config YAML -> tokenizer -> tokens -> treino -> amostra.

Amarra as peças de `src/tucanoce` num pipeline reprodutível e sem hiperparâmetro
hardcoded (tudo vem do YAML). Uso:

    python scripts/pretrain.py --config configs/medium.yaml
    python scripts/pretrain.py --config configs/medium.yaml --prompt "the electron"

Sem corpus em disco (data/corpus.jsonl), cai num pequeno corpus de física embutido
para o pipeline rodar offline de ponta a ponta. Para o corpus real, rode antes o
scraping (ver src/tucanoce/data/scrape.py) ou aponte --corpus para seu JSONL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

# permite rodar como `python scripts/pretrain.py` sem instalar o pacote
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tucanoce.config import load_configs
from tucanoce.data.dataset import TextDataset
from tucanoce.inference.generate import generate
from tucanoce.model.transformer import TucanoCE
from tucanoce.tokenizer.bpe import BPETokenizer
from tucanoce.training.train import train

EOT = "<|endoftext|>"

# Fallback offline: minicorpus de física (mesmo domínio do corpus inicial).
_FALLBACK_CORPUS = [
    "the electron is a fundamental particle with negative electric charge",
    "quantum mechanics describes the behavior of particles at atomic scales",
    "the energy of a photon is proportional to its frequency times planck constant",
    "in quantum field theory particles are excitations of underlying fields",
    "the mass of the electron is much smaller than the mass of the proton",
    "energy and mass are related by the famous equation e equals m c squared",
    "the electromagnetic field carries energy and momentum through empty space",
    "particles with half integer spin are called fermions and obey pauli exclusion",
    "the quantum state of a particle is described by its complex wavefunction",
    "conservation of energy is a fundamental principle of classical and modern physics",
    "general relativity describes gravity as the curvature of spacetime by mass",
    "quarks combine to form hadrons such as protons and neutrons via strong force",
] * 40


def load_corpus(corpus_path: str) -> list[str]:
    """Lê os textos do JSONL limpo; se não existir, usa o fallback embutido."""
    if os.path.exists(corpus_path):
        texts = []
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                texts.append(json.loads(line)["text"])
        print(f"corpus: {len(texts)} artigos de {corpus_path}")
        return texts
    print(f"corpus não encontrado em {corpus_path} -> usando fallback offline "
          f"({len(_FALLBACK_CORPUS)} docs)")
    return _FALLBACK_CORPUS


def get_tokenizer(texts: list[str], vocab_size: int, tok_path: str) -> BPETokenizer:
    """Carrega o tokenizer do disco ou treina e salva. O id (tamanho do vocab)
    entra na validação do cache de tokens."""
    if os.path.exists(tok_path):
        tok = BPETokenizer.load(tok_path)
        print(f"tokenizer: carregado de {tok_path} (vocab {tok.actual_vocab_size})")
        return tok
    print(f"tokenizer: treinando BPE (alvo vocab {vocab_size})...")
    tok = BPETokenizer(vocab_size=vocab_size)
    tok.train(texts)
    if os.path.dirname(tok_path):
        os.makedirs(os.path.dirname(tok_path), exist_ok=True)
    tok.save(tok_path)
    print(f"tokenizer: {tok.actual_vocab_size} tokens, {len(tok.merges)} merges -> {tok_path}")
    return tok


def load_or_cache_tokens(texts, tokenizer, cache_path, tokenizer_id, corpus_path=None):
    """Tokeniza o corpus num fluxo único (com EOT entre docs), cacheando em disco.

    Validação robusta (nb 05): o cache depende de DUAS coisas — o tokenizer E o
    conteúdo do corpus. Validamos o id do tokenizer E o mtime do corpus. Checar só
    um dos dois é a armadilha clássica do nb 05: re-limpei o corpus sem trocar o
    tokenizer -> mtime muda, cache invalida; retreinei o tokenizer sem mexer no
    corpus -> id muda, cache invalida. Os dois juntos fecham o buraco.
    """
    eot_id = tokenizer.special_ids.get(EOT, 0)
    if os.path.exists(cache_path):
        blob = torch.load(cache_path)
        tok_ok = blob.get("tokenizer_id") == tokenizer_id
        corpus_ok = (corpus_path is None or not os.path.exists(corpus_path)
                     or os.path.getmtime(corpus_path) <= os.path.getmtime(cache_path))
        if tok_ok and corpus_ok:
            print(f"tokens: cache hit ({blob['tokens'].numel()} tokens)")
            return blob["tokens"]
        motivo = "corpus mudou" if not corpus_ok else "tokenizer mudou"
        print(f"tokens: cache inválido ({motivo}) -> re-tokenizando")

    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text))
        ids.append(eot_id)
    tokens = torch.tensor(ids, dtype=torch.long)
    if os.path.dirname(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"tokens": tokens, "tokenizer_id": tokenizer_id}, cache_path)
    print(f"tokens: {tokens.numel()} tokens tokenizados e cacheados -> {cache_path}")
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/medium.yaml")
    ap.add_argument("--corpus", default=None, help="override do corpus_path do YAML")
    ap.add_argument("--prompt", default="the electron")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model_cfg, train_cfg, data_cfg = load_configs(args.config)
    corpus_path = args.corpus or data_cfg.get("corpus_path", "data/corpus.jsonl")
    tok_path = data_cfg.get("tokenizer_path", "data/tokenizer.json")
    cache_path = data_cfg.get("tokens_cache", "data/corpus_tokens.pt")
    stride = data_cfg.get("stride", None)

    # 1) corpus -> tokenizer -> tokens (com cache)
    texts = load_corpus(corpus_path)
    tokenizer = get_tokenizer(texts, model_cfg.vocab_size, tok_path)
    tokenizer_id = f"bpe-v{tokenizer.actual_vocab_size}"
    tokens = load_or_cache_tokens(texts, tokenizer, cache_path, tokenizer_id,
                                  corpus_path=corpus_path)

    # o vocab real do tokenizer manda no modelo (pode ser < alvo em corpus pequeno)
    model_cfg.vocab_size = tokenizer.actual_vocab_size

    # 2) datasets chunked (stride = context_len por padrão) + split treino/val
    split = int((1 - args.val_frac) * len(tokens))
    train_ds = TextDataset(tokens[:split], model_cfg.context_len, stride)
    val_ds = TextDataset(tokens[split:], model_cfg.context_len, stride)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("corpus pequeno demais p/ o context_len; reduza context_len "
                         "no YAML ou use um corpus maior.")
    train_ld = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=train_cfg.batch_size)
    print(f"dataset: train {len(train_ds)} | val {len(val_ds)} amostras "
          f"(context_len={model_cfg.context_len})")

    # 3) modelo + treino
    model = TucanoCE(model_cfg)
    print(f"modelo: {model.num_params():,} parâmetros "
          f"(d={model_cfg.embed_dim} L={model_cfg.n_layers} H={model_cfg.n_heads})")
    result = train(model, train_ld, val_ld, train_cfg, device=args.device)
    print(f"\ntreino concluído | best_val_loss {result['best_val_loss']:.4f} "
          f"(epoch {result['best_epoch']}) | checkpoint {result['ckpt_path']}")

    # 4) recarrega o melhor checkpoint e gera uma amostra
    ckpt = torch.load(result["ckpt_path"])
    model.load_state_dict(ckpt["model"])
    sample = generate(model, tokenizer, prompt=args.prompt, max_new_tokens=60,
                      temperature=0.8, top_k=40, repetition_penalty=1.2)
    print(f"\n--- amostra (prompt: {args.prompt!r}) ---\n{sample}\n")


if __name__ == "__main__":
    main()
