# `artigo/` — fonte versionada do artigo

O `.docx` na raiz é **gerado**, não editado à mão. A fonte de verdade é
`artigo.md`; editar o `.docx` diretamente faz a próxima geração descartar a
alteração.

## Regenerar

```bash
cd artigo
pandoc artigo.md \
  --from=markdown+tex_math_dollars+pipe_tables+simple_tables \
  --to=docx \
  --reference-doc=abnt_reference.docx \
  --toc --toc-depth=3 \
  --resource-path=.:media \
  -o ../TucanoCE_Artigo_ABNT.docx
```

O `--reference-doc` carrega os estilos ABNT (fontes, espaçamento, numeração de
títulos). `abnt_reference.docx` é só o `styles.xml` — 12 KB, sem conteúdo nem
imagens; serve exclusivamente como portador de estilo. As equações saem em OMML
nativo, editáveis no Word. O `--toc` recria o sumário.

## Figuras

- `media/rId*.png` — figuras derivadas dos *notebooks* (00–07).
- `../results/figuras/*.png` — figuras dos experimentos, **geradas**, não versionadas
  aqui em duplicata: o `artigo.md` as referencia por caminho relativo para que exista
  uma única cópia de cada binário no repositório.

## Números reportados

Nenhum número do artigo é digitado à mão duas vezes. Reproduzir:

```bash
python scripts/entropia_corpora.py          # Tabela 3 (proxies de entropia)
python scripts/ablacao_entropia_volume.py   # Tabela 5 e Figuras 17–18
python scripts/benchmark.py                 # Tabela 6 (BPB vs GPT-2)
python -m pytest tests -q                   # os 78 testes da Tabela 2
```

Os resultados ficam em `results/` como JSON, que é o insumo das figuras.
