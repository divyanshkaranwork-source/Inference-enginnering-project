"""Paged KV storage and the attention backend used by the batched engine.

The engine flattens every scheduled sequence into ONE token batch
("varlen" / "ragged" batching, as in vLLM and FlashAttention-varlen):

    seq A: 5 prefill tokens  |  seq B: 1 decode token  |  seq C: 1 decode token
    tokens  = [a0 a1 a2 a3 a4 | b7 | c3]           (T = 7)

All linear layers (QKV, O, FFN, LM head) run as one [T, D] x [D, *] matmul,
so every weight is read from memory once per step for *all* sequences -
that is where batching gets its throughput. Only attention is per-sequence,
because each sequence attends to its own KV blocks.

``slot_mapping[t] = block_table[pos // B] * B + pos % B`` tells the kernel
where token t's K/V vector lives in the pool - exactly vLLM's
``reshape_and_cache`` kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inferlab.config import LlamaConfig
from inferlab.foundations import ops

F32 = np.float32


class PagedKVPool:
    """K and V pools: [n_layers, num_blocks, block_size, n_kv_heads, head_dim]."""

    def __init__(self, cfg: LlamaConfig, num_blocks: int, block_size: int):
        shape = (cfg.n_layers, num_blocks, block_size, cfg.n_kv_heads, cfg.head_dim)
        self.k = np.zeros(shape, dtype=F32)
        self.v = np.zeros(shape, dtype=F32)
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.cfg = cfg

    @property
    def nbytes(self) -> int:
        return int(self.k.nbytes + self.v.nbytes)

    @staticmethod
    def blocks_for_budget(cfg: LlamaConfig, block_size: int, budget_bytes: int) -> int:
        per_block = 2 * cfg.n_layers * block_size * cfg.kv_dim * 4
        return max(int(budget_bytes // per_block), 1)


@dataclass
class SeqAttnMeta:
    query_start: int  # first row of this sequence in the flat batch
    query_len: int  # number of new tokens this step
    context_len: int  # tokens already in the KV cache before this step
    block_table: np.ndarray  # int32 [num_blocks_of_seq]
    needs_logits: bool  # last token of the chunk produces a sample


class PagedForwardContext:
    def __init__(self, pool: PagedKVPool, metas: list[SeqAttnMeta], kernels=None):
        self.pool = pool
        self.metas = metas
        self.kernels = kernels
        bs = pool.block_size
        pos, slots, logit_rows = [], [], []
        for m in metas:
            p = np.arange(m.context_len, m.context_len + m.query_len, dtype=np.int64)
            pos.append(p)
            slots.append(m.block_table[p // bs].astype(np.int64) * bs + p % bs)
            if m.needs_logits:
                logit_rows.append(m.query_start + m.query_len - 1)
        self.positions = np.concatenate(pos)
        self.slot_mapping = np.concatenate(slots)
        self.logit_indices = np.array(logit_rows, dtype=np.int64)

    def attend(self, layer: int, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        pool, bs = self.pool, self.pool.block_size
        KV, hd = pool.cfg.n_kv_heads, pool.cfg.head_dim
        # 1) scatter new K/V into their slots (vLLM: reshape_and_cache)
        kflat = pool.k[layer].reshape(-1, KV, hd)
        vflat = pool.v[layer].reshape(-1, KV, hd)
        kflat[self.slot_mapping] = k
        vflat[self.slot_mapping] = v
        # 2) per-sequence attention over its own blocks
        out = np.empty_like(q)
        for m in self.metas:
            total = m.context_len + m.query_len
            qs = slice(m.query_start, m.query_start + m.query_len)
            nb = -(-total // bs)
            table = m.block_table[:nb]
            if self.kernels is not None and m.query_len == 1:
                # decode: the C++ PagedAttention kernel reads blocks in place
                out[qs] = self.kernels.paged_attention_decode(
                    q[qs][0], pool.k[layer], pool.v[layer], table, total
                )[None]
                continue
            # gather this sequence's blocks into a contiguous view (copy)
            kk = pool.k[layer][table].reshape(-1, KV, hd)[:total]
            vv = pool.v[layer][table].reshape(-1, KV, hd)[:total]
            attn = self.kernels.attention if self.kernels is not None else ops.attention
            out[qs] = attn(q[qs], kk, vv, q_offset=m.context_len, causal=True)
        return out

    def finish(self) -> None:
        pass
