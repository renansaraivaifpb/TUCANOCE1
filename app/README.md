# `app/` — playground de chat (Streamlit)

- **`chat.py`** — interface web para conversar com o modelo treinado.

```bash
streamlit run app/chat.py
```

Escolha honesta de design: este é um **LM base** (só pré-treino, sem SFT/instruct).
Ele **continua texto**, não segue instruções nem responde perguntas — a UI de chat é
uma conveniência (cada turno alimenta o histórico como *prompt* e o modelo continua a
escrever). Comece um início de frase (ex.: *"Once upon a time"*) e veja como ele segue.

A barra lateral permite trocar checkpoint/tokenizer e ajustar os botões de amostragem
(temperatura, top-k, top-p, *repetition penalty*, seed). *Defaults* apontam para o
modelo treinado no TinyStories (`checkpoints/best.pt` + `data/tokenizer_tinystories.json`).

Requer o extra `app`: `uv pip install -e ".[app]"`.
