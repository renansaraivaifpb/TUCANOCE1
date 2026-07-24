# `model/` — o Transformer decoder-only

Derivado dos notebooks `03_arquitetura_base` (base GPT-2) e `04_modernizacao_llama`.

- **`transformer.py`** — `TucanoCE`, o modelo completo. *Forward*:
  `Embed → L blocos pre-norm → RMSNorm → logits`. Inclui:
  - **weight tying** (embedding de entrada = projeção de saída);
  - **init** `N(0, 0.02²)` com projeções residuais escaladas por `1/√(2L)` para
    manter a norma da rodovia residual em `O(1)`;
  - buffers de RoPE (cos/sin) pré-computados, não persistidos no checkpoint;
  - `forward(x, past_kvs, use_cache)`: sem cache retorna `logits`; com cache retorna
    `(logits, kvs)` para a inferência.
- **`block.py`** — o bloco *pre-norm*: `h = h + Attn(Norm(h)); h = h + MLP(Norm(h))`.
- **[`layers/`](layers/)** — as quatro peças LLaMA (RMSNorm, SwiGLU, RoPE, atenção).
