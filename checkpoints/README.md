# `checkpoints/` — pesos treinados

- **`best.pt`** — o melhor checkpoint (menor `val_loss`) salvo pelo treino. Artefato
  **gerado**, não versionado no Git (ver `.gitignore`); recrie com
  `python scripts/pretrain.py --config configs/<preset>.yaml`.

Formato (`torch.save`):

```python
{
  "model":    state_dict,   # pesos (chaves independem do nome da classe)
  "cfg":      dict,         # ModelConfig serializado — reconstrói a arquitetura
  "epoch":    int,
  "val_loss": float,
}
```

Como o `cfg` viaja junto, `generate.py`/`chat.py` reconstroem o modelo a partir do
próprio checkpoint — basta casar o **tokenizer** correspondente (tokenizer errado ⇒
texto-lixo). O `state_dict` usa nomes de parâmetros (`embed.weight`, `blocks.0…`),
então renomear a classe do modelo **não** quebra o carregamento.
