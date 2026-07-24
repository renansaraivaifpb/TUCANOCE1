# `scripts/` — entrypoints de linha de comando

Amarram as peças de `src/tucanoce/` em fluxos reprodutíveis. Todos aceitam `--config`
apontando para um YAML de [`configs/`](../configs/) (fonte única de hiperparâmetros).

| Script | O que faz |
|---|---|
| `pretrain.py` | Pipeline de pré-treino: config → BPE (treina/carrega) → cache de tokens → datasets chunked → treino → checkpoint → amostra. |
| `generate.py` | Carrega um checkpoint e **gera texto** de um *prompt*. A arquitetura vem do próprio checkpoint; o tokenizer, do YAML (têm de casar). |
| `benchmark.py` | Benchmark **honesto vs GPT-2** em *bits/byte* (métrica independente de tokenizer) sobre held-out do TinyStories, + amostras e velocidade. |
| `download_tinystories.py` | Baixa o TinyStories do HuggingFace e converte para o JSONL do projeto (com teto de histórias, p/ CPU). |
| `scrape_corpus.py` | Coleta um corpus de *machine learning* da Wikipedia (categorias densas) → JSONL limpo. |

Exemplos:

```bash
python scripts/download_tinystories.py --max-stories 8000
python scripts/pretrain.py  --config configs/tinystories_cpu.yaml
python scripts/generate.py  --config configs/tinystories_cpu.yaml --prompt "Once upon a time"
python scripts/benchmark.py --gpt2 gpt2 --n-eval 150
```
