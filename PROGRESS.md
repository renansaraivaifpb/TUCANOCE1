# PROGRESS — estado do projeto e próximos passos

> Fonte da verdade do andamento do TucanoCE. Atualize ao concluir cada etapa.
> Última atualização: 2026-07-23.

## Estado atual

### ✅ Concluído — camada educacional (notebooks)

Os 8 notebooks estão prontos, **executam limpos do zero** (`jupyter nbconvert --execute`),
cada `.ipynb` é regenerável a partir do seu builder em `notebooks/_builders/`, e todos
trazem **figuras matplotlib** embutidas (23 no total) ilustrando os conceitos geométricos
(paisagem de custo, heatmaps de atenção, rotação de RoPE, curvas de treino, Chinchilla, etc.).

| # | Notebook | Paper | Peça(s) de `src/` que alimenta |
|---|---|---|---|
| 00 | `00_fundamentos_nn.ipynb` | 3b1b 1–4 | (base conceitual: backprop, gradient descent) |
| 01 | `01_transformer_intuicao.ipynb` | §2 | `model/layers/attention.py`, `model/block.py` |
| 02 | `02_tokenizacao_bpe.ipynb` | §3 | `tokenizer/bpe.py` |
| 03 | `03_arquitetura_base.ipynb` | §4 | `model/block.py`, `model/transformer.py` |
| 04 | `04_modernizacao_llama.ipynb` | §5 | `model/layers/{rmsnorm,swiglu,rope,attention}.py` |
| 05 | `05_pipeline_dados.ipynb` | §6 | `data/scrape.py`, `data/dataset.py` |
| 06 | `06_treinamento.ipynb` | §7 | `training/train.py` |
| 07 | `07_avaliacao_scaling.ipynb` | §8 | `training/evaluate.py`, `inference/generate.py` |

### ✅ Concluído — infraestrutura

- `config.py` — `ModelConfig`/`TrainConfig` + presets `small`→`xl` (validado: `hidden(512)=1408`).
- `data/dataset.py` — `TextDataset` chunked já implementado (razão chunked/sliding ≈ 124×).
- Docs: `README.md`, `ARCHITECTURE.md` (o que herdei/troquei do blueprint de regressão).
- `configs/medium.yaml`, `pyproject.toml`, `.python-version`.

### ✅ Concluído — os stubs de `src/` (o LM completo)

Todos os stubs foram preenchidos derivando de cada notebook. `src/` agora é um LM
treinável de ponta a ponta, validado por um smoke test (tokenizer round-trip,
weight tying, init loss ≈ log V, **KV-cache idêntico ao recompute** com erro < 5e-7,
treino convergindo e geração):

| Peça | nb de origem | Verificação-chave |
|---|---|---|
| `tokenizer/bpe.py` (+ save/load) | 02 | round-trip perfeito byte-level |
| `layers/rmsnorm.py` | 04 | RMS em FP32 |
| `layers/swiglu.py` | 04 | gate·up→down |
| `layers/rope.py` | 04 | rotate_half, offset posicional |
| `layers/attention.py` | 04 | RoPE + KV-cache + is_causal adaptativo |
| `model/block.py` | 03 | pre-norm residual |
| `model/transformer.py` | 03 | init 1/√(2L), weight tying, buffers RoPE |
| `training/train.py` | 06 | AdamW 2 grupos, cosine+warmup, clip, accum, early stop |
| `training/evaluate.py` | 07 | val_loss + acc (soma/total) |
| `inference/generate.py` | 07+04 | sampling + KV-cache |
| `data/scrape.py` | 05 | limpeza componível + rate limit/Retry-After |

**Correção de bug herdado:** `num_params()` subtraía `V·d` indevidamente
(`parameters()` já deduplica o peso do weight tying). Agora bate com o paper:
medium **42,7M**, large 91,2M, xl 210,8M.

### ✅ Concluído — entrypoint e config

- `src/tucanoce/config.py::load_configs()` — lê YAML (única fonte da verdade).
- `configs/small.yaml` — preset dev/CPU (roda offline com corpus fallback).
- `scripts/pretrain.py` — pipeline: config → BPE (train/load) → cache de tokens
  (validado por id do tokenizer) → datasets chunked → treino → checkpoint → amostra.
  Testado offline: `python scripts/pretrain.py --config configs/small.yaml`.

### ✅ Concluído — testes formais (`tests/`, 78 passando)

Suíte pytest espelhando `src/` (`pyproject.toml`: `pythonpath=["src"]`), fixtures em
`tests/conftest.py`. Cobre invariâncias, não qualidade de texto. Ablações pedidas:
sliding vs chunked, RMSNorm em BF16, top-p. Rodar: `python -m pytest`.

### ✅ Concluído — corpus real + treino ponta-a-ponta (CPU)

- **Scraping**: `scripts/scrape_corpus.py` → **154 artigos de física de partículas**
  da Wikipedia inglesa. `scrape.py` refatorado: batching `exlimit=20` + subcategorias.
- **Curadoria (achado real)**: os extracts vazavam TeX/MathML (`{\displaystyle ...}`)
  — **28% do corpus era markup**, e o modelo aprendia a cuspir LaTeX. Corrigido em
  `clean_article._remove_math_markup`; corpus re-limpo (154 artigos, ~416K tokens).
- **BPE otimizado** (saída idêntica): train por frequência de palavra + encode
  minbpe com cache → encode do corpus **65s → 0,2s**.
- **Pipeline** `scripts/pretrain.py` (config→BPE→cache→treino→amostra); cache de
  tokens agora valida **mtime do corpus E id do tokenizer** (pitfall do nb 05).
- **Treino** (preset `small`, 1,8M params, corpus limpo): val_loss 4,19→**3,087**
  (piso log(4096)=8,32), acc 0,50; geração em inglês de física coerente, sem markup.
  T/N ≈ 416K/1,8M ≈ 0,23 → fortemente *data-constrained* (Chinchilla, nb 07).

## Próximos passos (ordem recomendada)

1. **Mais dados** (a alavanca real): recolher as demais categorias de física
   (o scrape parou na 1ª — `Particle_physics`); alvo ~10-100M tokens.
2. **Treinar o `medium`/`large` em GPU** (aqui é CPU-only): reproduzir os números do
   paper (val_loss ~2,13; BPC ~0,68 no corpus 2×) com `configs/medium.yaml`.
3. **Roadmap além do pré-treino** (paper §9, ver `ARCHITECTURE.md`): avaliação
   estruturada, Muon, GQA, SFT+LoRA. Achado central confirmado no nosso próprio
   treino: **o gargalo é dado, não capacidade**.

## Convenções do projeto (para não quebrar)

- **Notebooks são gerados**: edite `_builders/build_NN.py`, rode `python3 _builders/build_NN.py`,
  valide com `jupyter nbconvert --to notebook --execute --inplace NN_*.ipynb`. Nunca edite o `.ipynb` à mão.
- **Config centralizada**: hiperparâmetros vivem em `config.py`/`configs/*.yaml`, não hardcoded.
- **Path injetável** em todo I/O (default de produção + override em teste).
