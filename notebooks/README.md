# `notebooks/` — a camada educacional

Os 8 notebooks derivam cada peça do modelo em PyTorch, na ordem do fluxo, com a
matemática ao lado do código executável e figuras embutidas. São o núcleo didático
do projeto; o pacote `src/tucanoce/` é a versão de produção do que se deriva aqui.

| # | Notebook | Paper | Alimenta em `src/` |
|---|---|---|---|
| 00 | `00_fundamentos_nn` | 3b1b 1–4 | base conceitual (backprop, gradient descent) |
| 01 | `01_transformer_intuicao` | §2 | `model/layers/attention.py`, `model/block.py` |
| 02 | `02_tokenizacao_bpe` | §3 | `tokenizer/bpe.py` |
| 03 | `03_arquitetura_base` | §4 | `model/block.py`, `model/transformer.py` |
| 04 | `04_modernizacao_llama` | §5 | `model/layers/{rmsnorm,swiglu,rope,attention}.py` |
| 05 | `05_pipeline_dados` | §6 | `data/scrape.py`, `data/dataset.py` |
| 06 | `06_treinamento` | §7 | `training/train.py` |
| 07 | `07_avaliacao_scaling` | §8 | `training/evaluate.py`, `inference/generate.py` |

## Convenção importante: os notebooks são **gerados**

Não edite os `.ipynb` à mão. Cada notebook é produzido por um *builder* em
`_builders/build_NN.py` (o `.ipynb` é JSON frágil; o builder mantém o diff limpo).

```bash
python3 _builders/build_NN.py                                   # regenera o .ipynb (sem outputs)
jupyter nbconvert --to notebook --execute --inplace NN_*.ipynb  # executa e embute figuras
```

Todos executam limpos do zero.
