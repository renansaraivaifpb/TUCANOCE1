"""Tokenizer BPE byte-level estilo GPT-2 (paper seção 3, Algorithms 1 e 2).

Opera sobre bytes (0-255), não caracteres Unicode. Treina merges iterativos:
conta pares adjacentes mais frequentes e funde até atingir vocab_size.

Gargalos e otimizações (seção 3.2.2, 3.4):
- treino dos merges: O(n) scan por merge; aceitável em Python p/ corpus pequeno.
- encoding: reimplementado em C via ctypes em produção (~100x). Aqui, Python puro.
- cache em disco dos tokens (load_or_cache_corpus), invalidado por mtime.

Ver notebook 02_tokenizacao_bpe.ipynb.
"""
from __future__ import annotations

import json
import re
from collections import Counter


class BPETokenizer:
    # Pré-tokenização: separa palavras (\w+), espaços (\s+) e pontuação. A
    # concatenação dos chunks reconstrói o texto EXATO (partição sem perda), e
    # nunca fundimos pares que cruzam a fronteira de um chunk.
    PAT = re.compile(r"\w+|\s+|[^\w\s]+", re.UNICODE)

    def __init__(self, vocab_size: int = 8192,
                 special_tokens: tuple[str, ...] = ("<|endoftext|>",)):
        self.vocab_size = vocab_size
        self.special_tokens = tuple(special_tokens)
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}
        self.special_ids: dict[str, int] = {}
        self._encode_cache: dict[str, list[int]] = {}   # memo por palavra (Zipf)
        self._build_base_vocab()

    # ------------------------------------------------------------------ helpers
    def _build_base_vocab(self) -> None:
        # Vocab inicial = 256 bytes + tokens especiais. Cobertura total, sem <UNK>.
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_ids = {}
        for sp in self.special_tokens:
            new_id = len(self.vocab)
            self.vocab[new_id] = sp.encode("utf-8")
            self.special_ids[sp] = new_id

    def _pretokenize(self, text: str) -> list[str]:
        return self.PAT.findall(text)

    @staticmethod
    def _get_stats(chunk_ids: list[list[int]]) -> Counter:
        pairs: Counter = Counter()
        for ids in chunk_ids:
            for a, b in zip(ids, ids[1:]):
                pairs[(a, b)] += 1
        return pairs

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # -------------------------------------------------------------------- train
    def train(self, corpus: list[str]) -> None:
        """Algorithm 1: par mais frequente -> token novo, repetir até vocab_size.

        Otimização de produção (§3.2.2): em vez de manter a lista de TODOS os chunks
        (com repetições), colapsamos em contagens de palavras únicas e ponderamos as
        estatísticas de pares pela frequência. Como texto é Zipfiano (poucas palavras
        dominam), isso é ordens de magnitude mais rápido e produz merges IDÊNTICOS.
        """
        self._build_base_vocab()
        self.merges = {}
        self._encode_cache = {}
        text = "\n".join(corpus)
        word_freq = Counter(self._pretokenize(text))
        # cada palavra única -> (lista de ids de byte, frequência)
        words = [list(w.encode("utf-8")) for w in word_freq]
        freqs = list(word_freq.values())

        n_merges = self.vocab_size - len(self.vocab)
        for _ in range(n_merges):
            stats: Counter = Counter()
            for ids, fr in zip(words, freqs):
                for a, b in zip(ids, ids[1:]):
                    stats[(a, b)] += fr                 # pondera pela frequência da palavra
            if not stats:
                break                                   # corpus já totalmente fundido
            best = max(stats, key=stats.get)            # par mais frequente (ponderado)
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            self.merges[best] = new_id
            words = [self._merge(ids, best, new_id) for ids in words]

    # ------------------------------------------------------------ encode/decode
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._encode_cache.get(chunk)
        if cached is not None:                          # memo por palavra (Zipf)
            return cached
        ids = list(chunk.encode("utf-8"))
        # estilo minbpe: a cada passo aplica o merge de MENOR rank (new_id) presente
        # — equivale a aplicar os merges na ordem aprendida, mas visita só os pares
        # que existem na palavra (muito mais rápido que varrer todos os merges).
        while len(ids) >= 2:
            best_rank, best_pair = None, None
            for pair in zip(ids, ids[1:]):
                rank = self.merges.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break                                   # nenhum par é mergível
            ids = self._merge(ids, best_pair, best_rank)
        self._encode_cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        """Codifica texto -> ids. Tokens especiais são reconhecidos e não fundidos."""
        if not self.special_tokens:
            ids_out: list[int] = []
            for chunk in self._pretokenize(text):
                ids_out.extend(self._encode_chunk(chunk))
            return ids_out

        # Isola tokens especiais antes da pré-tokenização (ex.: <|endoftext|>).
        pattern = "(" + "|".join(re.escape(sp) for sp in self.special_tokens) + ")"
        ids_out = []
        for piece in re.split(pattern, text):
            if not piece:
                continue
            if piece in self.special_ids:
                ids_out.append(self.special_ids[piece])
            else:
                for chunk in self._pretokenize(piece):
                    ids_out.extend(self._encode_chunk(chunk))
        return ids_out

    def decode(self, ids: list[int]) -> str:
        # Sem perda: cada id -> sequência de bytes; concatena e decodifica UTF-8.
        b = b"".join(self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    # --------------------------------------------------------------- persistência
    def save(self, path: str) -> None:
        """Serializa merges + metadados em JSON. `vocab` é reconstruível a partir
        deles, então não precisa ser gravado."""
        payload = {
            "vocab_size": self.vocab_size,
            "special_tokens": list(self.special_tokens),
            # merges como lista ordenada [[a, b, new_id], ...] — JSON não tem tupla-chave
            "merges": [[a, b, nid] for (a, b), nid in self.merges.items()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls(vocab_size=payload["vocab_size"],
                  special_tokens=tuple(payload["special_tokens"]))
        # reaplica os merges na ordem para reconstruir merges + vocab
        for a, b, new_id in payload["merges"]:
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        return tok

    @property
    def actual_vocab_size(self) -> int:
        return len(self.vocab)
