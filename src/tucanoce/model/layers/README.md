# `model/layers/` — as modernizações do padrão LLaMA

As quatro peças que separam a base GPT-2 do padrão LLaMA. Derivadas do notebook
`04_modernizacao_llama`.

- **`rmsnorm.py`** — RMSNorm: normaliza por `x/√(mean(x²)+ε)` (sem centralizar média
  nem *bias*). Calculado em **FP32** mesmo sob BF16, por estabilidade numérica.
- **`swiglu.py`** — MLP com portão: `(SiLU(x·Wg) ⊙ (x·Wu))·Wd`, hidden `≈ 8d/3`
  arredondado a múltiplo de 64 (alinhamento com *tensor cores*).
- **`rope.py`** — codificação posicional rotacional: `precompute_rope_freqs`,
  `rotate_half`, `apply_rope`. A pontuação de atenção passa a depender só da posição
  **relativa** `n−m`.
- **`attention.py`** — atenção causal *multi-head* via `scaled_dot_product_attention`
  (Flash Attention quando disponível), com **RoPE** e **KV-cache**. Detalhe-chave:
  `is_causal = (|Q| == |K|)` — no *decode* com cache a query única vê todo o cache.
