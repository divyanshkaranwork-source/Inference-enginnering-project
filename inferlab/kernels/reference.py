"""NumPy reference implementations of the kernel *algorithms*.

These are slow on purpose: written loop-by-loop so you can read the
algorithm that FlashAttention / PagedAttention / online softmax implement in
CUDA. Tests check them against the simple versions in ``foundations.ops``.
"""

from __future__ import annotations

import numpy as np

F32 = np.float32


def online_softmax(x: np.ndarray) -> np.ndarray:
    """One pass for max + normaliser (Milakov & Gimelshein, 2018)."""
    m, s = -np.inf, 0.0
    for xi in x:
        if xi > m:
            s = s * np.exp(m - xi) + 1.0 if np.isfinite(m) else 1.0
            m = xi
        else:
            s += np.exp(xi - m)
    return np.exp(x - m) / s


def flash_attention(q, k, v, q_offset: int = 0, causal: bool = True, block_k: int = 32):
    """Tiled attention with online softmax. q[T,H,hd], k/v[S,KV,hd].

    Memory per (head, query): O(hd + block_k) instead of O(S).
    """
    T, H, hd = q.shape
    S, KV, _ = k.shape
    g = H // KV
    scale = 1.0 / np.sqrt(hd)
    out = np.zeros_like(q, dtype=np.float64)
    for h in range(H):
        kv = h // g
        for i in range(T):
            limit = min(S, q_offset + i + 1) if causal else S
            m, l, acc = -np.inf, 0.0, np.zeros(hd)
            for j0 in range(0, limit, block_k):
                j1 = min(limit, j0 + block_k)
                s = (k[j0:j1, kv] @ q[i, h]) * scale  # scores for this key tile
                m_new = max(m, float(s.max()))
                alpha = np.exp(m - m_new) if np.isfinite(m) else 0.0
                p = np.exp(s - m_new)
                l = l * alpha + p.sum()
                acc = acc * alpha + p @ v[j0:j1, kv]
                m = m_new
            out[i, h] = acc / l if l > 0 else 0.0
    return out.astype(F32)


def paged_attention_decode(q, k_cache, v_cache, block_table, context_len: int):
    """q[H,hd]; caches [num_blocks, block_size, KV, hd]; follow the block table."""
    H, hd = q.shape
    _, bs, KV, _ = k_cache.shape
    g = H // KV
    scale = 1.0 / np.sqrt(hd)
    out = np.zeros((H, hd))
    n_blocks = -(-context_len // bs)
    for h in range(H):
        kv = h // g
        m, l, acc = -np.inf, 0.0, np.zeros(hd)
        for b in range(n_blocks):
            phys = int(block_table[b])
            n = min(bs, context_len - b * bs)
            s = (k_cache[phys, :n, kv] @ q[h]) * scale
            m_new = max(m, float(s.max()))
            alpha = np.exp(m - m_new) if np.isfinite(m) else 0.0
            p = np.exp(s - m_new)
            l = l * alpha + p.sum()
            acc = acc * alpha + p @ v_cache[phys, :n, kv]
            m = m_new
        out[h] = acc / l
    return out.astype(F32)


def tiled_matmul(A, B, tile: int = 32):
    """Cache-blocked GEMM written out explicitly (C = A @ B)."""
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N), dtype=F32)
    for i0 in range(0, M, tile):
        for j0 in range(0, N, tile):
            for k0 in range(0, K, tile):
                a = A[i0 : i0 + tile, k0 : k0 + tile]
                b = B[k0 : k0 + tile, j0 : j0 + tile]
                C[i0 : i0 + tile, j0 : j0 + tile] += a @ b
    return C
