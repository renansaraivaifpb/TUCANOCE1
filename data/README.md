# `data/` — corpora e artefatos de dados

Artefatos **gerados** pelos scripts (não versionados no Git — ver `.gitignore`).
Recrie-os com `scripts/download_tinystories.py` / `scripts/scrape_corpus.py` e
`scripts/pretrain.py`.

| Arquivo | O que é | Gerado por |
|---|---|---|
| `corpus_tinystories.jsonl` | corpus TinyStories (histórias infantis) | `download_tinystories.py` |
| `corpus.jsonl` | corpus de *machine learning* (Wikipedia) | `scrape_corpus.py` |
| `corpus_physics.jsonl` | corpus de física (experimento original, backup) | `scrape_corpus.py` |
| `tokenizer_*.json` | tokenizer BPE treinado por corpus (merges + vocab) | `pretrain.py` |
| `corpus_tokens_*.pt` | cache do corpus tokenizado (fluxo de ids) | `pretrain.py` |

Formato do JSONL: uma linha por documento, `{"text", "title", "source", "category"}`.

O cache de tokens é validado por **mtime do corpus + id do tokenizer**: se qualquer
um mudar, é re-tokenizado automaticamente (evita treinar com cache obsoleto).
