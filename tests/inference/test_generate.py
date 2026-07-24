"""Testes de sampling e geração (nb 07) — cada botão + o loop com KV-cache."""
from __future__ import annotations

import torch

from tucanoce.config import ModelConfig
from tucanoce.inference.generate import generate, sample_next
from tucanoce.model.transformer import TucanoCE
from tucanoce.tokenizer.bpe import BPETokenizer

_LOGITS = torch.tensor([5.0, 4.0, 3.0, 1.0, 0.0, -2.0])


def test_temperature_zero_is_argmax():
    assert sample_next(_LOGITS, temperature=0.0) == 0    # maior logit


def test_top_k_one_is_deterministic():
    # com top_k=1 só o argmax sobrevive -> amostragem retorna sempre ele
    for _ in range(5):
        assert sample_next(_LOGITS, temperature=1.0, top_k=1) == 0


def test_top_p_keeps_at_least_one():
    # top_p minúsculo: só o token de maior massa sobrevive
    for _ in range(5):
        assert sample_next(_LOGITS, temperature=1.0, top_p=0.01) == 0


def test_repetition_penalty_demotes_seen_token():
    # penalidade forte derruba o token 0 (já visto) abaixo do token 1 -> muda o argmax
    out = sample_next(_LOGITS, temperature=1.0, top_k=1,
                      repetition_penalty=10.0, prev_tokens=[0])
    assert out == 1


def test_sample_next_respects_generator():
    # mesmo generator (seed) -> mesma amostra (reprodutibilidade)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = sample_next(_LOGITS, temperature=1.0, top_k=4, generator=g1)
    b = sample_next(_LOGITS, temperature=1.0, top_k=4, generator=g2)
    assert a == b


def _model_and_tok(toy_corpus):
    tok = BPETokenizer(vocab_size=300)
    tok.train(toy_corpus)
    cfg = ModelConfig(vocab_size=tok.actual_vocab_size, context_len=32,
                      n_layers=1, n_heads=2, embed_dim=16, dropout=0.0)
    return TucanoCE(cfg).eval(), tok


def test_generate_returns_str(toy_corpus):
    model, tok = _model_and_tok(toy_corpus)
    out = generate(model, tok, prompt="the electron", max_new_tokens=10,
                   device="cpu")
    assert isinstance(out, str) and out.startswith("the electron")


def test_generate_greedy_is_deterministic(toy_corpus):
    # temperatura 0 = argmax: duas gerações devem coincidir
    model, tok = _model_and_tok(toy_corpus)
    kw = dict(prompt="the", max_new_tokens=12, temperature=0.0, device="cpu")
    assert generate(model, tok, **kw) == generate(model, tok, **kw)


def test_generate_handles_context_overflow(toy_corpus):
    # context_len=32 + 40 tokens novos: o cache estoura a janela e o generate
    # deve reprocessar a janela recente sem estourar as tabelas RoPE (não crashar).
    model, tok = _model_and_tok(toy_corpus)
    out = generate(model, tok, prompt="energy", max_new_tokens=40,
                   temperature=0.8, top_k=20, device="cpu")
    assert isinstance(out, str)


def test_generate_empty_prompt(toy_corpus):
    model, tok = _model_and_tok(toy_corpus)
    out = generate(model, tok, prompt="", max_new_tokens=5, device="cpu")
    assert isinstance(out, str)
