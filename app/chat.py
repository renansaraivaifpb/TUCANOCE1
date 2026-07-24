"""Chat playground do TucanoCE — conversa com o LM que treinamos do zero.

Uso:
    streamlit run app/chat.py

Escolha honesta de design: este é um LM BASE (só pré-treino, sem SFT/instruct).
Ele NÃO segue instruções nem "responde perguntas" — ele CONTINUA texto. A UI de
chat é uma conveniência: cada turno alimenta o histórico como prompt e o modelo
continua a escrever. É um "playground de continuação" com cara de chat. Instruct
(SFT+LoRA) é o próximo passo do roadmap (ARCHITECTURE.md §9).
"""
from __future__ import annotations

import os
import sys

import streamlit as st
import torch

# permite `streamlit run app/chat.py` sem instalar o pacote
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tucanoce.config import ModelConfig
from tucanoce.inference.generate import generate_stream
from tucanoce.model.transformer import TucanoCE
from tucanoce.tokenizer.bpe import BPETokenizer

DEFAULT_CKPT = "checkpoints/best.pt"
DEFAULT_TOK = "data/tokenizer_tinystories.json"

st.set_page_config(page_title="TucanoCE · chat", page_icon="🦜", layout="centered")


@st.cache_resource(show_spinner="Carregando modelo e tokenizer...")
def load_model(ckpt_path: str, tok_path: str):
    """Carrega checkpoint + tokenizer uma vez (cache_resource evita recarregar a
    cada interação). Retorna (model, tokenizer, meta)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ModelConfig(**ckpt["cfg"])
    model = TucanoCE(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer = BPETokenizer.load(tok_path)
    meta = {"params": model.num_params(), "vocab": cfg.vocab_size,
            "ctx": cfg.context_len, "epoch": ckpt.get("epoch"),
            "val_loss": ckpt.get("val_loss")}
    return model, tokenizer, meta


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🦜 TucanoCE")
st.sidebar.caption("LM estilo LLaMA construído do zero")

ckpt_path = st.sidebar.text_input("Checkpoint", DEFAULT_CKPT)
tok_path = st.sidebar.text_input("Tokenizer", DEFAULT_TOK)

st.sidebar.subheader("Amostragem")
temperature = st.sidebar.slider("Temperatura", 0.0, 1.5, 0.8, 0.05,
                                help="0 = determinístico (argmax); alto = mais variado")
top_k = st.sidebar.slider("top-k", 0, 100, 40, 1, help="0 = desligado")
top_p = st.sidebar.slider("top-p (nucleus)", 0.0, 1.0, 0.0, 0.05, help="0 = desligado")
rep_pen = st.sidebar.slider("Repetition penalty", 1.0, 2.0, 1.2, 0.05,
                            help="> 1 combate loops")
max_new = st.sidebar.slider("Máx. tokens novos", 16, 256, 80, 8)
seed = st.sidebar.number_input("Seed (−1 = aleatório)", value=-1, step=1)

if st.sidebar.button("🧹 Limpar conversa"):
    st.session_state.messages = []
    st.rerun()

# ------------------------------------------------------------------ carga
if not os.path.exists(ckpt_path):
    st.error(f"Checkpoint não encontrado: `{ckpt_path}`.\n\n"
             "Treine primeiro:\n\n"
             "```\npython scripts/pretrain.py --config configs/tinystories_cpu.yaml\n```")
    st.stop()

model, tokenizer, meta = load_model(ckpt_path, tok_path)

st.title("Chat com o TucanoCE")
vl = f"{meta['val_loss']:.3f}" if meta["val_loss"] is not None else "?"
st.caption(f"{meta['params']:,} params · vocab {meta['vocab']} · contexto "
           f"{meta['ctx']} · melhor val_loss {vl} (epoch {meta['epoch']})")
st.info("Modelo **base** (só pré-treino em histórias infantis do TinyStories): ele "
        "**continua texto**, não segue instruções. Comece um início de história e "
        "veja como ele segue. Ex.: *'Once upon a time'*, *'The little girl'*.", icon="ℹ️")

# ------------------------------------------------------------------ chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def build_prompt(messages, context_len, tokenizer) -> str:
    """Concatena o histórico num único prompt e recorta ao contexto do modelo
    (em tokens), preservando o final — o que o modelo realmente vê."""
    text = "\n".join(m["content"] for m in messages)
    ids = tokenizer.encode(text)
    if len(ids) > context_len:
        text = tokenizer.decode(ids[-context_len:])
    return text


if user_input := st.chat_input("Escreva um início de história..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    prompt = build_prompt(st.session_state.messages, meta["ctx"], tokenizer)
    gen = None if seed < 0 else torch.Generator().manual_seed(int(seed))

    with st.chat_message("assistant"):
        stream = generate_stream(
            model, tokenizer, prompt, max_new_tokens=int(max_new),
            temperature=float(temperature),
            top_k=int(top_k) if top_k > 0 else None,
            top_p=float(top_p) if top_p > 0 else None,
            repetition_penalty=float(rep_pen), device="cpu", generator=gen,
        )
        reply = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": reply})
