"""Helper mínimo para gerar notebooks .ipynb programaticamente.

Por que gerar por código em vez de editar .ipynb à mão: o formato .ipynb é JSON
com metadados de execução embutidos; editar direto é frágil e polui o diff. Aqui
descrevemos o notebook como uma lista de células e deixamos o nbformat serializar.

Uso:
    from _nbbuild import md, code, build
    build("00_fundamentos.ipynb", [
        md("# Título\\n\\ntexto..."),
        code("import torch\\nprint(torch.__version__)"),
    ])
"""
from __future__ import annotations

import os
import nbformat as nbf


def md(source: str) -> nbf.NotebookNode:
    """Célula markdown."""
    return nbf.v4.new_markdown_cell(source.strip("\n"))


def code(source: str) -> nbf.NotebookNode:
    """Célula de código (sem outputs — o usuário executa)."""
    return nbf.v4.new_code_cell(source.strip("\n"))


def build(filename: str, cells: list, kernel: str = "python3") -> str:
    """Monta e grava o notebook ao lado deste helper. Retorna o caminho."""
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": kernel},
        "language_info": {"name": "python"},
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    return out
