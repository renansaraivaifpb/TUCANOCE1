"""Testes da limpeza de artigos (nb 05). Scraping em si (rede) não é testado aqui."""
from __future__ import annotations

from tucanoce.data.scrape import clean_article

_BODY = ("Quarks are elementary particles and a fundamental constituent of matter. "
         "There are six flavors: up, down, charm, strange, top, and bottom. Quarks "
         "combine to form hadrons such as protons and neutrons, and owing to color "
         "confinement they are never directly observed in isolation in any experiment. "
         "The quark model was independently proposed by Murray Gell-Mann and George "
         "Zweig in 1964, and has since been confirmed by many scattering experiments.")


def test_strips_trailing_sections():
    dirty = _BODY + "\n\n== References ==\n[1] foo\n== External links ==\nbar\n"
    clean = clean_article(dirty)
    assert clean is not None
    assert "References" not in clean and "External links" not in clean
    assert "bar" not in clean


def test_removes_ref_markers():
    dirty = "Quarks[1] combine[2] into hadrons[34]. " + _BODY
    clean = clean_article(dirty)
    assert "[1]" not in clean and "[2]" not in clean and "[34]" not in clean


def test_collapses_blank_lines():
    dirty = _BODY + "\n\n\n\n\n" + _BODY
    clean = clean_article(dirty)
    assert "\n\n\n" not in clean


def test_stub_returns_none():
    assert clean_article("too short") is None
    assert clean_article("x" * 299) is None
    assert clean_article("x" * 300) is not None


def test_graceful_when_no_sections():
    # sem seção-lixo: degradação graciosa (não quebra, retorna o corpo limpo)
    assert clean_article(_BODY) is not None


def test_removes_math_markup():
    # curadoria: TeX {\displaystyle ...} + árvore MathML de chars soltos precisam sumir
    dirty = (_BODY + " charge conjugation \n  \n    \n        C\n      \n    "
             "{\\displaystyle C} and parity {\\displaystyle P_{\\mu \\nu }} here.")
    clean = clean_article(dirty)
    assert clean is not None
    assert "\\displaystyle" not in clean
    assert "{" not in clean and "}" not in clean
    # a prosa em volta permanece
    assert "charge conjugation" in clean and "parity" in clean
