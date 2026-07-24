# TucanoCE — um LM estilo LLaMA construído do zero

Reimplementação didática, do zero e peça por peça, de um **modelo de linguagem
autoregressivo** (decoder-only): parte do GPT-2 (2019) e moderniza para o padrão
LLaMA — **RMSNorm, SwiGLU, RoPE e KV-cache**. Treinável em CPU e em GPU de consumo
(RTX 5070, 12 GB). O nome vem do tucano + CE (Ceará).

Dupla natureza:

- **Camada educacional** — 8 *notebooks* numerados (`notebooks/00`→`07`) que derivam
  cada componente em PyTorch, com a matemática ao lado do código executável.
- **Código de produção** — o pacote `src/tucanoce/`, um LM completo e treinável de
  ponta a ponta, organizado por estágio de *pipeline* e coberto por **78 testes**.

## Estrutura do repositório

| Pasta | Conteúdo | README |
|---|---|---|
| [`src/tucanoce/`](src/tucanoce/) | o LM: tokenizer → data → model → training → inference | [↗](src/tucanoce/README.md) |
| [`notebooks/`](notebooks/) | 00→07, a camada educacional (geradas por `_builders/`) | [↗](notebooks/README.md) |
| [`scripts/`](scripts/) | entrypoints: treino, geração, benchmark, coleta de corpus | [↗](scripts/README.md) |
| [`configs/`](configs/) | presets de treino em YAML (fonte única de hiperparâmetros) | [↗](configs/README.md) |
| [`tests/`](tests/) | suíte pytest (invariâncias) espelhando `src/` | [↗](tests/README.md) |
| [`app/`](app/) | *playground* de chat em Streamlit | [↗](app/README.md) |
| [`data/`](data/) | corpora, tokenizers e caches de tokens (artefatos) | [↗](data/README.md) |
| [`checkpoints/`](checkpoints/) | pesos treinados (`best.pt`) | [↗](checkpoints/README.md) |

Documentos: [`ARCHITECTURE.md`](ARCHITECTURE.md) (decisões de arquitetura),
[`PROGRESS.md`](PROGRESS.md) (estado e próximos passos) e
`TucanoCE_Artigo_ABNT.docx` (artigo em formato ABNT com o benchmark).

## Instalação

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[notebooks,dev]"   # extras: notebooks, dev (pytest/ruff), app (streamlit)
```

## Uso rápido

```bash
# 1) baixar um corpus (TinyStories — recomendado para o modelo pequeno)
python scripts/download_tinystories.py --max-stories 8000

# 2) treinar (CPU, preset small)
python scripts/pretrain.py --config configs/tinystories_cpu.yaml

# 3) gerar texto a partir do checkpoint
python scripts/generate.py --config configs/tinystories_cpu.yaml --prompt "Once upon a time"

# 4) chat interativo (LM base: continua texto, não segue instruções)
streamlit run app/chat.py

# 5) benchmark honesto vs GPT-2 (métrica de bits/byte)
python scripts/benchmark.py --gpt2 gpt2
```

## Achado central

O gargalo de qualidade é **dado, não capacidade** (regime Chinchilla) — e, em modelo
pequeno, o que manda é **casar a entropia do corpus com a capacidade do modelo**. O
mesmo modelo de 1,8M params teve val_loss 3,09 (física) → 3,02 (ML) → **1,59
(TinyStories)** sem qualquer mudança de arquitetura: só trocando para texto de menor
entropia o modelo passou a gerar frases coerentes.

## Créditos

Baseado num paper de referência de construção de LM (não incluído no repositório) e
nas transcrições da série *Deep Learning* do 3Blue1Brown. As seções (§2–§8) citadas
no código/notebooks remetem a esse paper.
