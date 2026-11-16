"""Core transformer operators in plain NumPy.

Every function here is a readable reference for a kernel that production
engines implement in CUDA / Triton / Metal / AVX. Shapes use these letters:

    T  number of query tokens in the batch
    S  number of key/value tokens (context length)
    H  query heads,  KV  key/value heads,  hd  head dimension
"""

from __future__ import annotations

import numpy as np

F32 = np.float32


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight   (Llama uses no bias/mean).

    Memory-bound: reads x once, writes y once -> production engines fuse it
    with the residual add ("fused add + RMSNorm") to save one HBM round trip.
    """
    x = x.astype(F32, copy=False)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps)) * weight


def silu(x: np.ndarray) -> np.ndarray:
    # clip the exponent to avoid overflow warnings for very negative inputs
    return x / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """SwiGLU activation used by Llama's FFN: silu(x W1) * (x W3)."""
    return silu(gate) * up


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (subtract the max before exp)."""
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(x - m)
    s = np.sum(e, axis=axis, keepdims=True)
    return e / np.where(s == 0, 1.0, s)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return x - m - np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))


# ---------------------------------------------------------------------------
# Rotary position embeddings (RoPE)
# ---------------------------------------------------------------------------
def rope_frequencies(head_dim: int, theta: float = 10000.0) -> np.ndarray:
    """inv_freq[i] = theta^(-2i/hd) for i in [0, hd/2)."""
    return (1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))).astype(F32)


def apply_rope(
    x: np.ndarray, positions: np.ndarray, theta: float = 10000.0, style: str = "interleaved"
) -> np.ndarray:
    """Rotate query/key vectors by an angle proportional to their position.

    x: [T, n_heads, hd], positions: [T]. Returns a new array.

    RoPE encodes *relative* position: <rope(q, m), rope(k, n)> depends only on
    (m - n). That is why the KV cache can store already-rotated keys and never
    touch them again.
    """
    T, _, hd = x.shape
    inv_freq = rope_frequencies(hd, theta)
    ang = positions.astype(F32)[:, None] * inv_freq[None, :]  # [T, hd/2]
    cos = np.cos(ang)[:, None, :]
    sin = np.sin(ang)[:, None, :]
    if style == "interleaved":  # pairs (0,1), (2,3), ...
        x0, x1 = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = x0 * cos - x1 * sin
        out[..., 1::2] = x0 * sin + x1 * cos
        return out
    if style == "half":  # pairs (i, i + hd/2), Hugging Face "rotate_half"
        h = hd // 2
        x0, x1 = x[..., :h], x[..., h:]
        return np.concatenate([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)
    raise ValueError(f"unknown rope style {style!r}")


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
def attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    q_offset: int = 0,
    causal: bool = True,
) -> np.ndarray:
    """Scaled dot-product attention with grouped-query attention (GQA).

    q: [T, H, hd]   k, v: [S, KV, hd]   ->   out: [T, H, hd]

    Query token i sits at absolute position ``q_offset + i`` and may attend to
    keys at positions <= q_offset + i (causal). ``q_offset = S - T`` is the
    usual case: the last T tokens of the context are the new queries.

    This *materialises* the full [H, T, S] score matrix - O(T*S) memory. See
    ``inferlab.kernels.reference.flash_attention`` for the tiled version that
    never does.
    """
    T, H, hd = q.shape
    S, KV, _ = k.shape
    g = H // KV
    scale = 1.0 / np.sqrt(hd)
    # [KV, g, T, hd] @ [KV, 1, hd, S] -> [KV, g, T, S]
    qh = q.reshape(T, KV, g, hd).transpose(1, 2, 0, 3)
    kh = k.transpose(1, 2, 0)[:, None, :, :]
    scores = (qh @ kh) * scale
    if causal:
        qpos = q_offset + np.arange(T)[:, None]
        kpos = np.arange(S)[None, :]
        scores = np.where(kpos <= qpos, scores, -np.inf)
    probs = softmax(scores, axis=-1)
    vh = v.transpose(1, 0, 2)[:, None, :, :]  # [KV, 1, S, hd]
    out = probs @ vh  # [KV, g, T, hd]
    return out.transpose(2, 0, 1, 3).reshape(T, H, hd).astype(F32, copy=False)
