# `tokenizer/` — tokenização BPE byte-level

Converte texto ↔ sequência de ids. Derivado do notebook `02_tokenizacao_bpe`.

- **`bpe.py`** — `BPETokenizer`: *Byte-Pair Encoding* em nível de *byte* (parte dos
  256 bytes e funde iterativamente o par adjacente mais frequente). O artefato
  versionado são as **regras de fusão** (*merges*) — o equivalente, num LM, à
  "engenharia de atributos".

Pontos de implementação:

- **Round-trip perfeito**: `decode(encode(x)) == x` para qualquer texto (byte-level).
- **Treino por frequência de palavra** + `encode` com cache (estilo *minbpe*): o
  encode do corpus inteiro cai de dezenas de segundos para frações de segundo.
- **Tokens especiais** (ex.: `<|endoftext|>`) com ids reservados.
- **`save`/`load`** em JSON — o `vocab_size` real entra na validação do cache de
  tokens e define o `vocab_size` do modelo.
