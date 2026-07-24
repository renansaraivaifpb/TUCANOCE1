# `training/` — treino e avaliação

Derivado dos notebooks `06_treinamento` (treino) e `07_avaliacao_scaling` (métricas).

- **`train.py`** — o laço de treino. Decisões:
  - **AdamW com dois grupos de parâmetros**: *weight decay* em matrizes (2D+),
    decaimento zero em vetores 1D (*biases*, ganhos de normalização).
  - **Scheduler**: *warmup* linear + decaimento por cosseno até `0.1·lr_max`.
  - **Grad clipping** por norma global e **grad accumulation** para *batch* efetivo.
  - **BF16** (em GPU) e **early stopping** com *patience*, salvando o melhor checkpoint.
- **`evaluate.py`** — `val_loss` (cross-entropy, acumulada por soma) e *accuracy* de
  próximo token; helper `bits_per_char` (BPC), comparável entre *tokenizers*.

Referência: o piso trivial de uma cross-entropy é `log(V)` (chute uniforme); a
primeira checagem de sanidade de qualquer treino é a *loss* inicial ≈ `log V`.
