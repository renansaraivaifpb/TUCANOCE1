"""Scraping de artigos da Wikipedia em inglês (nb 05).

Corpus em inglês (não português) de propósito: densidade muito maior em física
técnica, o que testa melhor a arquitetura em domínio denso.

Cada artigo vira uma linha JSONL: {text, title, source, category}.

Robustez (nb 05): rate limit >= 1.5s entre requisições; respeitar o
cabeçalho Retry-After em HTTP 429 em vez de backoff exponencial cego.

Limpeza (nb 05): remover References/External links/See also/Further reading;
remover marcadores [1],[2]; colapsar quebras de linha; descartar stubs (<300 chars).

Ver notebook 05_pipeline_dados.ipynb.
"""
from __future__ import annotations

import json
import os
import re
import time

WIKI_API = "https://en.wikipedia.org/w/api.php"
MIN_INTERVAL = 1.5   # segundos entre requisições (nb 05)

# Seções cujo TÍTULO exato marca o fim do conteúdo útil (nb 05).
DROP_SECTIONS = ("References", "External links", "See also", "Further reading",
                 "Notes", "Bibliography")


# ---------------------------------------------------------------------- limpeza
# Funções puras e componíveis: cada uma faz uma coisa e é testável isolada. Se
# uma seção não existe no artigo, a função simplesmente não faz nada (degradação
# graciosa) em vez de quebrar.

def _strip_trailing_sections(text: str) -> str:
    """Corta o texto a partir do primeiro cabeçalho '== Título ==' sem valor."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        heading = line.strip().strip("=").strip()       # '== References ==' -> 'References'
        if line.strip().startswith("==") and heading in DROP_SECTIONS:
            return "\n".join(lines[:i])
    return text                                          # nada a cortar


def _remove_ref_markers(text: str) -> str:
    """Remove marcadores numéricos de citação: [1], [12], [3][4]..."""
    return re.sub(r"\[\d+\]", "", text)


def _collapse_blank_lines(text: str) -> str:
    """Colapsa 3+ quebras de linha consecutivas em exatamente 2."""
    return re.sub(r"\n{3,}", "\n\n", text)


def _remove_math_markup(text: str) -> str:
    """Remove o TeX/MathML que os extracts da Wikipedia deixam vazar das fórmulas.

    Curadoria de dados (nb 05): tags <math> vêm como blocos `{\\displaystyle ...}`
    precedidos por uma "árvore" MathML de caracteres soltos, um por linha, muito
    indentada. Sem isso o modelo aprende a cuspir LaTeX. Achado ao inspecionar as
    gerações — metade do corpus estava contaminada.
    """
    # 1) remove blocos {\displaystyle ...} com chaves BALANCEADAS (há aninhamento)
    marker = "{\\displaystyle"
    out, i = [], 0
    while i < len(text):
        j = text.find(marker, i)
        if j == -1:
            out.append(text[i:])
            break
        out.append(text[i:j])
        depth, k = 0, j
        while k < len(text):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    k += 1
                    break
            k += 1
        i = k
    text = "".join(out)
    # 2) descarta as linhas-lixo da árvore MathML (1-2 chars não-espaço). Em extracts
    #    de prosa as linhas são parágrafos longos ou títulos '== ... ==', nunca de
    #    1-2 chars — então isso é seguro.
    lines = [ln for ln in text.split("\n") if not (0 < len(ln.strip()) <= 2)]
    return "\n".join(lines)


def clean_article(text: str, min_chars: int = 300) -> str | None:
    """Pipeline de limpeza. Retorna None para stubs (curtos demais após limpar)."""
    text = _strip_trailing_sections(text)
    text = _remove_ref_markers(text)
    text = _remove_math_markup(text)
    text = _collapse_blank_lines(text)
    text = text.strip()
    return text if len(text) >= min_chars else None


# --------------------------------------------------------------------- scraping

def _request_json(session, url: str, params: dict) -> dict:
    """GET com rate limit + Retry-After. `session` guarda o timestamp da última
    chamada em session['last'] (mutável, sem variável global)."""
    wait = MIN_INTERVAL - (time.monotonic() - session["last"])
    if wait > 0:
        time.sleep(wait)

    for _ in range(5):
        resp = session["get"](url, params=params, timeout=30,
                              headers={"User-Agent": "tucanoce-scraper/0.1 (educational)"})
        session["last"] = time.monotonic()
        if resp.status_code == 429:
            # respeita Retry-After (o servidor diz exatamente quanto esperar)
            retry_after = int(resp.headers.get("Retry-After", 30))
            time.sleep(max(retry_after, MIN_INTERVAL))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"429 persistente após 5 tentativas: {url}")


def _category_members(session, category: str, cmtype: str, limit: int) -> list[str]:
    """Lista membros de uma categoria (cmtype='page' ou 'subcat'), paginando."""
    titles: list[str] = []
    cmcontinue = None
    while len(titles) < limit:
        params = {
            "action": "query", "format": "json", "list": "categorymembers",
            "cmtitle": f"Category:{category}", "cmtype": cmtype,
            "cmlimit": min(500, limit - len(titles)),
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = _request_json(session, WIKI_API, params)
        titles.extend(m["title"] for m in data["query"]["categorymembers"])
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return titles[:limit]


def _collect_titles(session, category: str, limit: int, include_subcats: bool) -> list[str]:
    """Junta títulos de artigos da categoria e, opcionalmente, de suas subcategorias
    diretas (1 nível). Categorias de física costumam ter poucos artigos diretos e
    muitas subcategorias — descer 1 nível aumenta a cobertura sem explodir o crawl."""
    titles = _category_members(session, category, "page", limit)
    if include_subcats and len(titles) < limit:
        subcats = _category_members(session, category, "subcat", 20)
        for sub in subcats:
            if len(titles) >= limit:
                break
            name = sub.split(":", 1)[-1]            # 'Category:Foo' -> 'Foo'
            titles.extend(_category_members(session, name, "page", limit - len(titles)))
    return titles[:limit]


# extracts aceita até 20 títulos por requisição (exlimit): ~20x menos chamadas que
# 1-a-1, a otimização de produção descrita no nb 05.
_EXLIMIT = 20


def _articles_text_batch(session, titles: list[str]) -> dict[str, str]:
    """Baixa o texto plano de até _EXLIMIT artigos numa requisição. Retorna
    {title: extract}. Segue 'continue' (extracts pode devolver os textos em partes)."""
    out: dict[str, str] = {}
    base = {
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": "true", "exlimit": _EXLIMIT, "redirects": "1",
        "titles": "|".join(titles),
    }
    cont: dict = {}
    while True:
        data = _request_json(session, WIKI_API, {**base, **cont})
        for page in data.get("query", {}).get("pages", {}).values():
            extract = page.get("extract")
            if extract:                              # acumula (pode vir em partes)
                out[page["title"]] = out.get(page["title"], "") + extract
        if "continue" in data:
            cont = data["continue"]
        else:
            break
    return out


def scrape_categories(categories: list[str], per_category: int = 500,
                      out_path: str = "data/corpus.jsonl",
                      include_subcats: bool = True, verbose: bool = True) -> int:
    """Coleta artigos das categorias e grava um JSONL limpo. Retorna nº de artigos.

    Deduplica por título (categorias de física se sobrepõem muito) e busca os
    textos em lotes de _EXLIMIT (menos requisições, menos risco de 429). Requer o
    pacote `requests`. Cada artigo vira uma linha:
        {"text": ..., "title": ..., "source": "wikipedia", "category": ...}
    """
    import requests   # import tardio: só o scraping real depende de rede/requests

    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    session = {"get": requests.get, "last": 0.0}
    seen: set[str] = set()
    n_written = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for category in categories:
            titles = _collect_titles(session, category, per_category, include_subcats)
            # só títulos inéditos, preservando ordem
            fresh = [t for t in titles if not (t in seen or seen.add(t))]
            cat_written = 0
            for i in range(0, len(fresh), _EXLIMIT):
                batch = fresh[i:i + _EXLIMIT]
                extracts = _articles_text_batch(session, batch)
                for title in batch:
                    text = clean_article(extracts.get(title, "") or "")
                    if text is None:                    # stub/ausente descartado
                        continue
                    f.write(json.dumps({
                        "text": text, "title": title,
                        "source": "wikipedia", "category": category,
                    }, ensure_ascii=False) + "\n")
                    n_written += 1
                    cat_written += 1
            if verbose:
                print(f"  [{category}] {len(fresh)} títulos -> {cat_written} artigos "
                      f"(total {n_written})", flush=True)
    return n_written
