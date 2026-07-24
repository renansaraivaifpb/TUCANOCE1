"""Geração autoregressiva com KV-cache (paper seção 5.4).

Sem cache, cada passo refaz o forward inteiro: O(L*d^2*N*(p+N)), quadrático em N.
Com cache, K,V das posições já processadas são reaproveitados (a máscara causal
garante que não mudam): O(L*d^2*N), linear em N (Eq. 51 vs 52).

Detalhes:
- offset posicional em RoPE: token novo no passo k está na posição global p+k;
  usa cos/sin dessa posição, não da posição 0 (Eq. 54). Tratado na atenção.
- máscara: no prefill Q e K têm mesmo T (is_causal=True); no decode Q tem 1 token
  e "vê" todo o cache sem máscara (Listing 4).

Sampling (seção 2.8): temperatura, top-k, top-p, repetition penalty.

Ver notebook 07_avaliacao_scaling.ipynb (uso) e 04 (mecânica do cache).
"""
from __future__ import annotations

from collections.abc import Iterator

import torch


def sample_next(logits, temperature: float = 1.0, top_k: int | None = None,
                top_p: float | None = None, repetition_penalty: float = 1.0,
                prev_tokens=None, generator=None) -> int:
    """logits: (vocab,). Retorna id amostrado (int).

    Ordem (importa): repetition penalty -> temperatura -> top-k -> top-p -> softmax.
    """
    logits = logits.clone().float()

    # repetition penalty (estilo CTRL): empurra tokens já emitidos para baixo,
    # combatendo loops. Divide se logit>0, multiplica se <0 (sempre reduz a massa).
    if repetition_penalty != 1.0 and prev_tokens is not None:
        for t in set(prev_tokens):
            if logits[t] > 0:
                logits[t] /= repetition_penalty
            else:
                logits[t] *= repetition_penalty

    if temperature <= 0:                                 # 0 = argmax determinístico
        return int(logits.argmax())
    logits = logits / temperature                        # afia (<1) ou achata (>1)

    if top_k is not None:                                 # zera tudo fora dos k maiores
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[-1]
        logits[logits < kth] = float("-inf")

    if top_p is not None:                                 # nucleus: menor conjunto com massa >= p
        s_logits, s_idx = torch.sort(logits, descending=True)
        cum = torch.softmax(s_logits, dim=-1).cumsum(dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()       # mantém sempre o 1º token
        remove[..., 0] = False
        logits[s_idx[remove]] = float("-inf")

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 100,
             temperature: float = 0.8, top_k: int | None = 40,
             top_p: float | None = None, repetition_penalty: float = 1.0,
             device=None, generator=None) -> str:
    """Gera texto a partir de um prompt, com KV-cache.

    Faz um prefill do prompt inteiro (1 forward), depois decodifica 1 token por
    passo reaproveitando K,V. Quando o cache atinge context_len (limite das tabelas
    RoPE pré-computadas), reprocessa a janela — RoPE é relativo, então reposicionar
    não muda a semântica; só volta ao custo O(N) de um prefill pontual.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    context_len = model.cfg.context_len

    ids = tokenizer.encode(prompt)
    if not ids:                                          # prompt vazio: parte do nada
        ids = [0]

    # --- prefill: forward do prompt (recortado à janela) ---
    x = torch.tensor([ids[-context_len:]], dtype=torch.long, device=device)
    logits, past = model(x, use_cache=True)
    next_logits = logits[0, -1]

    for _ in range(max_new_tokens):
        nxt = sample_next(next_logits, temperature, top_k, top_p,
                          repetition_penalty, prev_tokens=ids, generator=generator)
        ids.append(nxt)

        cached_len = past[0][0].size(-2)                 # comprimento atual do cache K
        if cached_len >= context_len:
            # cache cheio: reprocessa a janela mais recente (RoPE é relativo)
            x = torch.tensor([ids[-context_len:]], dtype=torch.long, device=device)
            logits, past = model(x, use_cache=True)
        else:
            # decode incremental: só o token novo entra no forward
            x = torch.tensor([[nxt]], dtype=torch.long, device=device)
            logits, past = model(x, past_kvs=past, use_cache=True)
        next_logits = logits[0, -1]

    return tokenizer.decode(ids)


@torch.no_grad()
def generate_stream(model, tokenizer, prompt: str, max_new_tokens: int = 100,
                    temperature: float = 0.8, top_k: int | None = 40,
                    top_p: float | None = None, repetition_penalty: float = 1.0,
                    device=None, generator=None) -> Iterator[str]:
    """Igual a generate(), mas é um GERADOR que emite o texto NOVO em pedaços — para
    o efeito de digitação num chat (st.write_stream).

    Decodificação incremental correta: tokens byte-level podem partir um caractere
    UTF-8 multibyte no meio. Acumulamos os bytes num buffer e só emitimos os
    caracteres já completos, retendo a cauda incompleta até o próximo token fechá-la.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    context_len = model.cfg.context_len

    ids = tokenizer.encode(prompt) or [0]
    x = torch.tensor([ids[-context_len:]], dtype=torch.long, device=device)
    logits, past = model(x, use_cache=True)
    next_logits = logits[0, -1]

    buf = b""
    for _ in range(max_new_tokens):
        nxt = sample_next(next_logits, temperature, top_k, top_p,
                          repetition_penalty, prev_tokens=ids, generator=generator)
        ids.append(nxt)

        buf += tokenizer.vocab[nxt]
        try:                                             # emite só chars completos
            text = buf.decode("utf-8")
            buf = b""
        except UnicodeDecodeError as e:                  # retém a cauda multibyte
            text = buf[:e.start].decode("utf-8")
            buf = buf[e.start:]
        if text:
            yield text

        cached_len = past[0][0].size(-2)
        if cached_len >= context_len:
            x = torch.tensor([ids[-context_len:]], dtype=torch.long, device=device)
            logits, past = model(x, use_cache=True)
        else:
            x = torch.tensor([[nxt]], dtype=torch.long, device=device)
            logits, past = model(x, past_kvs=past, use_cache=True)
        next_logits = logits[0, -1]

    if buf:                                              # flush de bytes residuais
        yield buf.decode("utf-8", errors="replace")
