# Arquitetura do TucanoCE (adaptação do blueprint para um LM)

O `ARCHITECTURE_BLUEPRINT.md` na raiz veio de um projeto de **regressão tabular**
(XGBoost). A filosofia transfere; a estrutura literal, não. Este documento registra
o que foi herdado, o que foi trocado, e por quê.

## O que herdamos do blueprint

1. **`src/` organizado por estágio de pipeline**, não por tipo de arquivo. O código
   espelha o fluxo mental do problema. Aqui os estágios são:
   `tokenizer → data → model → training → inference`.
2. **Cada estágio roda isolado** e é importável/testável sozinho.
3. **Camada de config real** (`config.py` + `configs/*.yaml`), não defaults
   hardcoded — corrigindo a crítica nº1 da seção 7 do blueprint.
4. **Path injetável** para I/O (dataset, cache de tokens, checkpoints).

## O que trocamos (e por quê)

| Blueprint (regressão) | TucanoCE (LM) | Razão |
|---|---|---|
| `feature_engineering.py` + encoders `.pkl` | `tokenizer/bpe.py` | Num LM a "feature engineering" é a tokenização. O artefato versionado é o tokenizer (merges), não encoders de coluna. |
| Split temporal (train <2020…) | `stride` no `TextDataset` | Não há eixo temporal; a decisão de dados é sliding vs chunked (nb 05). |
| `train.py` vs `tune.py` (Optuna) | `train.py` com early stopping | Em pré-treino de LM não se faz HPO caro por trial; o loop já traz save-best + patience (nb 06). |
| Schema alignment (`reindex`) | contexto fixo `context_len` | O "contrato" de entrada é o comprimento de contexto e o vocab_size, não colunas. |
| `.pkl` para modelo | checkpoint `torch` + tokenizer versionado | Pickle é frágil (crítica nº2); modelo salva state_dict; tokenizer salva merges. |
| batch mensal | geração autoregressiva (`inference/generate.py`) | Inferência de LM é geração token-a-token com KV-cache, não scoring em lote. |

## Estrutura

```
tucanoce/
├── notebooks/            # 00→07, exploração NUMERADA na ordem do fluxo (o núcleo educacional)
├── src/tucanoce/
│   ├── config.py         # ModelConfig + TrainConfig + presets (fonte da verdade)
│   ├── tokenizer/bpe.py  # BPE byte-level (nb 02)
│   ├── data/
│   │   ├── scrape.py     # Wikipedia → JSONL (nb 05)
│   │   └── dataset.py    # TextDataset chunked (nb 05)
│   ├── model/
│   │   ├── layers/       # rmsnorm, swiglu, rope, attention (nb 04)
│   │   ├── block.py      # bloco pre-norm (nb 03)
│   │   └── transformer.py# modelo completo + weight tying + init (nb 03)
│   ├── training/
│   │   ├── train.py      # AdamW, scheduler, BF16, early stopping (nb 06)
│   │   └── evaluate.py   # loss, accuracy, BPC (nb 07)
│   └── inference/
│       └── generate.py   # KV-cache + sampling (nb 04)
├── configs/medium.yaml   # config real (não vazia — crítica nº1 do blueprint)
├── tests/                # espelham src/
├── pyproject.toml
└── .python-version
```

## Roadmap (fora do escopo do pré-treino)

Ordenado por retorno esperado — e o achado central manda começar por dados:

1. **Mais dados** (prioridade alta): expandir corpus para ~100M tokens.
2. **Avaliação estruturada**: HellaSwag, ARC, MMLU (baseline vs literatura).
3. **Muon optimizer**, **GQA** (barateia KV-cache), **SFT + LoRA** (vira assistente).
4. **Quantização** (GPTQ/AWQ) — só faz sentido com modelo já bem treinado.
