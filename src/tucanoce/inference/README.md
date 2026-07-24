# `inference/` — geração autoregressiva

Derivado dos notebooks `07_avaliacao_scaling` (uso) e `04_modernizacao_llama` (cache).

- **`generate.py`**:
  - `sample_next` — aplica os botões de amostragem na ordem correta:
    *repetition penalty* → temperatura → *top-k* → *top-p* → softmax.
  - `generate` — *prefill* do prompt (1 *forward*) e depois *decode* de 1 token por
    passo reaproveitando o **KV-cache**: custo `O(N)` em vez de `O(N²)`. Quando o
    cache atinge `context_len`, reprocessa a janela (RoPE é relativo, então
    reposicionar não muda a semântica).
  - `generate_stream` — versão geradora que emite o texto em pedaços (para o efeito
    de digitação no chat), tratando corretamente caracteres UTF-8 multibyte.

> É um LM **base**: continua texto, não segue instruções. Instruct (SFT+LoRA) é
> passo futuro do roadmap.
