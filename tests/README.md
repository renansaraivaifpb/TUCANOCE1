# `tests/` — suíte de testes (pytest)

**78 testes** que cobrem **invariâncias** do modelo (não a qualidade do texto
gerado). A estrutura espelha `src/tucanoce/`; as *fixtures* comuns ficam em
`conftest.py`. O `pyproject.toml` define `pythonpath=["src"]`, então os testes
importam `tucanoce...` sem instalar o pacote.

```bash
python -m pytest          # roda tudo
python -m pytest -q tests/model/layers/test_rope.py   # um arquivo
```

Exemplos do que é verificado: round-trip do tokenizer; RMS calculado em FP32;
`rotate_half`/offset do RoPE; **KV-cache idêntico ao recompute** (erro < 5e-7);
*weight tying* (mesmo tensor na memória); *loss* inicial ≈ `log V`; ablações
pedidas (sliding vs chunked, RMSNorm em BF16, top-p).

Subpastas (`model/`, `model/layers/`, `data/`, `tokenizer/`, `training/`,
`inference/`) espelham os módulos correspondentes de `src/`.
