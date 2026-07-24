# `tucanoce/` — o modelo de linguagem

O pacote é organizado **por estágio de *pipeline***, não por tipo de arquivo: a
estrutura espelha o fluxo do problema. Cada estágio é importável e testável
isoladamente.

```
tokenizer → data → model → training → inference
```

| Módulo | Papel | README |
|---|---|---|
| `config.py` | `ModelConfig` + `TrainConfig` + presets; **fonte única de hiperparâmetros** e leitor de YAML (`load_configs`) | — |
| [`tokenizer/`](tokenizer/) | BPE byte-level (texto ↔ ids) | [↗](tokenizer/README.md) |
| [`data/`](data/) | coleta (Wikipedia) e `TextDataset` chunked | [↗](data/README.md) |
| [`model/`](model/) | o Transformer decoder-only (`TucanoCE`) + camadas LLaMA | [↗](model/README.md) |
| [`training/`](training/) | laço de treino (AdamW, scheduler, early stopping) e avaliação | [↗](training/README.md) |
| [`inference/`](inference/) | geração autoregressiva com KV-cache e sampling | [↗](inference/README.md) |

## Princípios de projeto

- **Config centralizada.** Nenhum hiperparâmetro *hardcoded*: tudo vem de `config.py`
  / `configs/*.yaml`. Corrige o antipadrão de *defaults* espalhados pelo código.
- **Path injetável.** Todo I/O (dataset, cache de tokens, checkpoint) recebe o
  caminho por parâmetro — *default* de produção, *override* em teste.
- **Responsabilidades separadas.** Funções puras e componíveis; cada peça faz uma
  coisa e é coberta por um teste em [`../../tests/`](../../tests/).

A classe principal é `tucanoce.model.transformer.TucanoCE`. O *forward* é
`Embed → L blocos pre-norm → RMSNorm → logits` com *weight tying*.
