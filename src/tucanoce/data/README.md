# `data/` — coleta e preparação do corpus

Derivado do notebook `05_pipeline_dados`.

- **`scrape.py`** — coleta artigos da Wikipedia (inglês) e grava JSONL limpo.
  - Robustez: *rate limit* ≥ 1,5 s entre requisições e respeito ao cabeçalho
    `Retry-After` em HTTP 429; *extracts* em lotes de 20 títulos.
  - Limpeza componível (`clean_article`): remove seções finais (References etc.),
    marcadores `[1]`, colapsa linhas em branco e — crucial — **`_remove_math_markup`**
    tira o TeX/MathML que vaza dos artigos (sem isso o modelo aprende a cuspir LaTeX).
- **`dataset.py`** — `TextDataset`: quebra o fluxo de tokens em janelas de
  `context_len`. O `stride` controla o modo: `None` ⇒ **chunked** (sem sobreposição,
  ~`context_len`× menos amostras que *sliding*, sem perda mensurável de qualidade).

Cada artigo é uma linha JSONL: `{"text", "title", "source", "category"}`.
