"""Sequence / context parallelism for long prompts: Ring Attention and Ulysses.

A 1M-token prompt does not fit one GPU's activations, so split the
*sequence* across ``cp`` GPUs.

Ring Attention (Liu et al. 2023): rank i keeps its query chunk Q_i and passes
K/V chunks around a ring. After cp steps every Q_i has seen every K/V chunk;
partial results are merged with the online-softmax (log-sum-exp) rule - the
same math as FlashAttention tiles, but the tiles live on different GPUs.
Communication (send/recv of K/V) overlaps with compute.

Causal masking makes contiguous chunks unbalanced (rank 0 has almost nothing
to attend to, the last rank everything). **Zig-zag** partitioning gives rank
i chunks i and 2cp-1-i, which balances the work.

Ulysses (DeepSpeed): two all-to-alls switch the sharding from sequence to
*heads*; each rank then runs ordinary attention for H/cp heads over the full
sequence. Simple, but cp is limited by the number of KV heads.
"""

from __future__ import annotations

import numpy as np

from inferlab.foundations import ops
from inferlab.parallel.cluster import SimCluster

F32 = np.float32


def partition(T: int, cp: int, scheme: str = "contiguous") -> list[np.ndarray]:
    if scheme == "contiguous":
        return list(np.array_split(np.arange(T), cp))
    if scheme == "zigzag":
        chunks = np.array_split(np.arange(T), 2 * cp)
        return [np.concatenate([chunks[i], chunks[2 * cp - 1 - i]]) for i in range(cp)]
    raise ValueError("scheme must be 'contiguous' or 'zigzag'")


def _partial(q, k, v, qpos, kpos):
    """Unnormalised attention of q against one K/V chunk: (acc, row_max, row_sum)."""
    H, hd = q.shape[1], q.shape[2]
    g = H // k.shape[1]
    kk = np.repeat(k, g, axis=1)
    vv = np.repeat(v, g, axis=1)
    s = np.einsum("thd,shd->hts", q, kk) / np.sqrt(hd)
    s = np.where(kpos[None, None, :] <= qpos[None, :, None], s, -np.inf)
    m = s.max(axis=-1)  # [H, Tq]
    m_safe = np.where(np.isfinite(m), m, 0.0)
    p = np.exp(s - m_safe[..., None])
    p = np.where(np.isfinite(s), p, 0.0)
    return np.einsum("hts,shd->thd", p, vv), m, p.sum(axis=-1)


def ring_attention(q, k, v, cp: int, scheme: str = "contiguous", cluster: SimCluster | None = None):
    """Causal ring attention. q[T,H,hd], k/v[T,KV,hd]. Returns (out, stats)."""
    T = q.shape[0]
    if not 1 <= cp <= T:
        raise ValueError("cp must be in [1, T]")
    cl = cluster or SimCluster(cp)
    parts = partition(T, cp, scheme)
    Q = [q[p] for p in parts]
    KV = [(k[p], v[p], p) for p in parts]  # what each rank currently holds
    H = q.shape[1]
    acc = [np.zeros((len(p), H, q.shape[2])) for p in parts]
    m = [np.full((H, len(p)), -np.inf) for p in parts]
    l = [np.zeros((H, len(p))) for p in parts]
    work = np.zeros(cp)  # unmasked score entries computed per rank (load balance)
    for step in range(cp):
        for r in range(cp):
            kc, vc, kp = KV[r]
            a, mc, lc = _partial(Q[r], kc, vc, parts[r], kp)
            work[r] += float(np.sum(kp[None, :] <= parts[r][:, None]))
            m_new = np.maximum(m[r], mc)
            m_new_safe = np.where(np.isfinite(m_new), m_new, 0.0)
            alpha = np.where(np.isfinite(m[r]), np.exp(m[r] - m_new_safe), 0.0)
            beta = np.where(np.isfinite(mc), np.exp(mc - m_new_safe), 0.0)
            acc[r] = acc[r] * alpha.T[..., None] + a * beta.T[..., None]
            l[r] = l[r] * alpha + lc * beta
            m[r] = m_new
        if step < cp - 1:  # pass K/V to the next rank in the ring
            KV = [
                (cl.send_recv(KV[(r - 1) % cp][0], "ring.k"), cl.send_recv(KV[(r - 1) % cp][1], "ring.v"),
                 KV[(r - 1) % cp][2])
                for r in range(cp)
            ]
    out = np.zeros_like(q, dtype=np.float64)
    for r, p in enumerate(parts):
        out[p] = acc[r] / l[r].T[..., None]
    stats = {
        "scheme": scheme,
        "work_per_rank": work.tolist(),
        "load_imbalance": float(work.max() / max(work.mean(), 1e-9)),
        "comm": cl.stats(),
    }
    return out.astype(F32), stats


def ulysses_attention(q, k, v, cp: int, cluster: SimCluster | None = None):
    """DeepSpeed-Ulysses: all-to-all sequence-shard -> head-shard -> attention -> back."""
    T, H, hd = q.shape
    KVH = k.shape[1]
    if H % cp or KVH % cp or T % cp:
        raise ValueError("Ulysses needs T, n_heads and n_kv_heads divisible by cp")
    cl = cluster or SimCluster(cp)
    seq = np.array_split(np.arange(T), cp)
    hq, hk = H // cp, KVH // cp

    def to_heads(x, hper):
        # rank i holds x[seq_i, :, :]; send head-group j to rank j
        send = [[x[seq[i]][:, j * hper : (j + 1) * hper] for j in range(cp)] for i in range(cp)]
        recv = cl.all_to_all(send, tag="ulysses.scatter_heads")
        return [np.concatenate(recv[j], axis=0) for j in range(cp)]  # [T, hper, hd] per rank

    qh, kh, vh = to_heads(q, hq), to_heads(k, hk), to_heads(v, hk)
    oh = [ops.attention(qh[j], kh[j], vh[j], 0, True) for j in range(cp)]
    send = [[oh[j][seq[i]] for i in range(cp)] for j in range(cp)]
    recv = cl.all_to_all(send, tag="ulysses.gather_seq")
    out = np.concatenate([np.concatenate(recv[i], axis=1) for i in range(cp)], axis=0)
    return out.astype(F32), {"comm": cl.stats()}
