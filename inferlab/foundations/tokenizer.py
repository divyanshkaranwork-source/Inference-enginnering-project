"""Tokenizers: text <-> token ids.

``Llama2Tokenizer`` reads llama2.c's ``tokenizer.bin`` - an export of the
**Llama-2 SentencePiece vocabulary (32000 tokens)**, i.e. the exact vocabulary
used by Llama-2-7B. Encoding is score-ordered BPE with UTF-8 byte fallback.

``ByteTokenizer`` is a trivial 259-token tokenizer (3 specials + 256 bytes)
used by the random demo model and the tests.

Two serving lessons live here:

1. **Special-token injection.** User text must never be able to produce
   control tokens (BOS/EOS/...). Our BPE never merges into special ids, so
   typing "</s>" in a prompt yields ordinary text tokens.
2. **Streaming detokenization.** A single emoji can span several byte tokens.
   ``StreamDecoder`` buffers incomplete UTF-8 so a stream never emits broken
   characters - every production server needs this.
"""

from __future__ import annotations

import codecs
import heapq
import os
import struct
from pathlib import Path


class TokenizerFormatError(ValueError):
    pass


class BaseTokenizer:
    bos_id: int = 1
    eos_id: int = 2
    vocab_size: int
    special_ids: frozenset[int]

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> list[int]:
        raise NotImplementedError

    def token_bytes(self, token_id: int, prev_id: int | None) -> bytes:
        raise NotImplementedError

    def id_to_piece(self, token_id: int) -> str:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        out = bytearray()
        prev = self.bos_id
        for t in ids:
            if t in self.special_ids:
                prev = t
                continue
            out += self.token_bytes(t, prev)
            prev = t
        return out.decode("utf-8", errors="replace")

    def stream_decoder(self) -> StreamDecoder:
        return StreamDecoder(self)


class StreamDecoder:
    """Incremental detokenizer: feed one token id at a time, get printable text."""

    def __init__(self, tok: BaseTokenizer):
        self.tok = tok
        self.prev: int | None = tok.bos_id
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, token_id: int) -> str:
        if token_id in self.tok.special_ids:
            self.prev = token_id
            return ""
        b = self.tok.token_bytes(token_id, self.prev)
        self.prev = token_id
        return self._dec.decode(b, final=False)

    def flush(self) -> str:
        return self._dec.decode(b"", final=True)


# ---------------------------------------------------------------------------
class ByteTokenizer(BaseTokenizer):
    """0=<unk>, 1=<s>, 2=</s>, 3..258 = raw bytes 0..255."""

    OFFSET = 3

    def __init__(self) -> None:
        self.vocab_size = 256 + self.OFFSET
        self.special_ids = frozenset({0, 1, 2})

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> list[int]:
        ids = [b + self.OFFSET for b in text.encode("utf-8")]
        return ([self.bos_id] if bos else []) + ids + ([self.eos_id] if eos else [])

    def token_bytes(self, token_id: int, prev_id: int | None) -> bytes:
        if not self.OFFSET <= token_id < self.vocab_size:
            return b""
        return bytes([token_id - self.OFFSET])

    def id_to_piece(self, token_id: int) -> str:
        if token_id in (0, 1, 2):
            return ("<unk>", "<s>", "</s>")[token_id]
        b = token_id - self.OFFSET
        ch = chr(b)
        return ch if 32 <= b < 127 else f"<0x{b:02X}>"


# ---------------------------------------------------------------------------
class Llama2Tokenizer(BaseTokenizer):
    def __init__(self, vocab: list[bytes], scores: list[float]):
        if len(vocab) != len(scores) or len(vocab) < 259:
            raise TokenizerFormatError("vocab/scores mismatch or vocabulary too small")
        self.vocab = vocab
        self.scores = scores
        self.vocab_size = len(vocab)
        # 0=<unk>, 1=<s>, 2=</s> in the Llama-2 vocabulary
        self.special_ids = frozenset({0, 1, 2})
        self.lookup: dict[bytes, int] = {}
        for i, piece in enumerate(vocab):
            if i in self.special_ids:
                continue  # never let user text BPE-merge into a control token
            self.lookup.setdefault(piece, i)
        self.space_id = self.lookup.get(b" ")
        # byte-fallback tokens <0x00>..<0xFF> live at ids 3..258
        self._byte_ids = {}
        for b in range(256):
            tid = self.lookup.get(f"<0x{b:02X}>".encode())
            self._byte_ids[b] = tid if tid is not None else 3 + b

    @classmethod
    def from_file(cls, path: str | os.PathLike, vocab_size: int = 32000) -> Llama2Tokenizer:
        data = Path(path).read_bytes()
        if len(data) < 4:
            raise TokenizerFormatError("tokenizer file too small")
        (max_len,) = struct.unpack_from("<i", data, 0)
        if not 0 < max_len <= 1024:
            raise TokenizerFormatError("invalid max token length")
        off, vocab, scores = 4, [], []
        for _ in range(vocab_size):
            if off + 8 > len(data):
                raise TokenizerFormatError("truncated tokenizer file")
            score, n = struct.unpack_from("<fi", data, off)
            off += 8
            if not 0 <= n <= max_len or off + n > len(data):
                raise TokenizerFormatError("invalid token length")
            vocab.append(data[off : off + n])
            scores.append(score)
            off += n
        return cls(vocab, scores)

    def id_to_piece(self, token_id: int) -> str:
        if not 0 <= token_id < self.vocab_size:
            return "<invalid>"
        if token_id in self.special_ids:
            return ("<unk>", "<s>", "</s>")[token_id]
        return self.vocab[token_id].decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int, prev_id: int | None) -> bytes:
        if not 0 <= token_id < self.vocab_size or token_id in self.special_ids:
            return b""
        piece = self.vocab[token_id]
        # SentencePiece adds a dummy-prefix space; drop it right after BOS
        if prev_id == self.bos_id and piece.startswith(b" "):
            piece = piece[1:]
        # raw byte tokens look like b"<0x0A>"
        if len(piece) == 6 and piece.startswith(b"<0x") and piece.endswith(b">"):
            try:
                return bytes([int(piece[3:5], 16)])
            except ValueError:
                return piece
        return piece

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> list[int]:
        """Score-ordered BPE using a heap - O(n log n) instead of llama2.c's O(n^2)."""
        ids: list[int] = []
        if text and self.space_id is not None:
            ids.append(self.space_id)  # SentencePiece "add_dummy_prefix"
        for ch in text:
            b = ch.encode("utf-8")
            tid = self.lookup.get(b)
            if tid is not None:
                ids.append(tid)
            else:
                ids.extend(self._byte_ids[x] for x in b)  # byte fallback
        ids = self._merge(ids)
        return ([self.bos_id] if bos else []) + ids + ([self.eos_id] if eos else [])

    def _merge(self, ids: list[int]) -> list[int]:
        n = len(ids)
        if n < 2:
            return ids
        pieces = [self.vocab[i] for i in ids]
        tok = list(ids)
        nxt = list(range(1, n)) + [-1]
        prv = [-1] + list(range(n - 1))
        alive = [True] * n
        heap: list[tuple[float, int, bytes, bytes]] = []

        def push(i: int) -> None:
            j = nxt[i]
            if j == -1:
                return
            cand = self.lookup.get(pieces[i] + pieces[j])
            if cand is not None:
                # max score first; ties -> leftmost (matches llama2.c)
                heapq.heappush(heap, (-self.scores[cand], i, pieces[i], pieces[j]))

        for i in range(n - 1):
            push(i)
        while heap:
            _, i, left, right = heapq.heappop(heap)
            j = nxt[i]
            # skip stale heap entries
            if not alive[i] or j == -1 or pieces[i] != left or pieces[j] != right:
                continue
            merged = left + right
            pieces[i], tok[i] = merged, self.lookup[merged]
            alive[j] = False
            nxt[i] = nxt[j]
            if nxt[j] != -1:
                prv[nxt[j]] = i
            push(i)
            if prv[i] != -1:
                push(prv[i])
        return [tok[i] for i in range(n) if alive[i]]


def load_tokenizer(path: str | os.PathLike | None, vocab_size: int) -> BaseTokenizer:
    if path is None:
        if vocab_size != 259:
            raise TokenizerFormatError("no tokenizer file given and vocab is not byte-level (259)")
        return ByteTokenizer()
    return Llama2Tokenizer.from_file(path, vocab_size)
