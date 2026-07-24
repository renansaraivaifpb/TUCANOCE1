"""Testes do BPETokenizer (nb 02) — round-trip, compressão, especiais, persistência."""
from __future__ import annotations

from tucanoce.tokenizer.bpe import BPETokenizer


def _trained(corpus, vocab_size=350):
    tok = BPETokenizer(vocab_size=vocab_size)
    tok.train(corpus)
    return tok


def test_roundtrip_ascii(toy_corpus):
    tok = _trained(toy_corpus)
    for t in ["the electron", "quantum field theory", "energy"]:
        assert tok.decode(tok.encode(t)) == t


def test_roundtrip_multibyte(toy_corpus):
    # garantia central do byte-level: acento e símbolo multibyte sobrevivem
    tok = _trained(toy_corpus)
    for t in ["é uma equação", "photon ⚛ spin", "😀 emoji"]:
        assert tok.decode(tok.encode(t)) == t


def test_base_vocab_has_256_bytes_plus_special(toy_corpus):
    tok = BPETokenizer(vocab_size=300)
    # antes de treinar: 256 bytes + 1 especial
    assert tok.actual_vocab_size == 257
    assert tok.special_ids["<|endoftext|>"] == 256


def test_compression_reduces_tokens(toy_corpus):
    tok = _trained(toy_corpus)
    text = " ".join(toy_corpus)
    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(tok.encode(text))
    assert n_tokens < n_bytes                 # BPE comprime vs byte puro
    assert n_bytes / n_tokens > 1.0


def test_vocab_size_respected(toy_corpus):
    tok = _trained(toy_corpus, vocab_size=300)
    assert tok.actual_vocab_size <= 300


def test_special_token_encodes_as_single_id(toy_corpus):
    tok = _trained(toy_corpus)
    ids = tok.encode("the electron<|endoftext|>quantum")
    assert tok.special_ids["<|endoftext|>"] in ids
    # e o round-trip preserva o especial
    assert tok.decode(ids) == "the electron<|endoftext|>quantum"


def test_save_load_roundtrip(toy_corpus, tmp_path):
    tok = _trained(toy_corpus)
    p = tmp_path / "tok.json"
    tok.save(str(p))
    tok2 = BPETokenizer.load(str(p))
    assert tok2.merges == tok.merges
    assert tok2.encode("quantum electron energy") == tok.encode("quantum electron energy")


def test_empty_and_no_special():
    tok = BPETokenizer(vocab_size=300, special_tokens=())
    tok.train(["abc abc abc"])
    assert tok.encode("") == []
    assert tok.decode(tok.encode("abc")) == "abc"
