"""A complete Llama-2 forward pass in NumPy.

    tokens -> embedding -> [ RMSNorm -> Attention(RoPE, GQA, KV cache) -> +residual
                             RMSNorm -> SwiGLU FFN                   -> +residual ] x L
           -> RMSNorm -> LM head -> logits

The model is deliberately agnostic of *how* keys/values are stored. It asks a
``ForwardContext`` for token positions and calls ``ctx.attend(layer, q, k, v)``.

* ``KVCache`` (below) is a contiguous per-sequence cache - what you write in
  a first inference engine and what llama.cpp does per slot.
* ``inferlab.runtime.paged.PagedForwardContext`` stores KV in fixed-size
  blocks shared by many sequences - vLLM's PagedAttention.

That split (model code vs. attention backend / KV layout) is exactly how
vLLM, SGLang and TensorRT-LLM are structured.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from inferlab.config import LlamaConfig
from inferlab.foundations import ops
from inferlab.foundations.quantization import LinearLike, make_linear

F32 = np.float32


@dataclass
class LayerWeights:
    attn_norm: np.ndarray  # [D]
    wq: LinearLike  # [H*hd, D]
    wk: LinearLike  # [KV*hd, D]
    wv: LinearLike  # [KV*hd, D]
    wo: LinearLike  # [D, H*hd]
    ffn_norm: np.ndarray  # [D]
    w1: LinearLike  # gate [F, D]
    w2: LinearLike  # down [D, F]
    w3: LinearLike  # up   [F, D]


@dataclass
class LlamaWeights:
    tok_embeddings: np.ndarray  # [V, D]
    layers: list[LayerWeights]
    final_norm: np.ndarray  # [D]
    lm_head: LinearLike  # [V, D]
    tied: bool = False  # lm_head shares memory with tok_embeddings

    def nbytes(self) -> int:
        total = self.tok_embeddings.nbytes + self.final_norm.nbytes
        if not self.tied:
            total += self.lm_head.nbytes
        for lw in self.layers:
            total += lw.attn_norm.nbytes + lw.ffn_norm.nbytes
            total += sum(m.nbytes for m in (lw.wq, lw.wk, lw.wv, lw.wo, lw.w1, lw.w2, lw.w3))
        return int(total)


class ForwardContext(Protocol):
    """What the model needs from the KV-cache / attention backend."""

    positions: np.ndarray  # [T] absolute positions of the tokens being processed
    logit_indices: np.ndarray | None  # rows to compute logits for (None = all rows)

    def attend(self, layer: int, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray: ...

    def finish(self) -> None: ...


# ---------------------------------------------------------------------------
# Contiguous KV cache (one sequence)
# ---------------------------------------------------------------------------
class KVCache:
    """K/V tensors of shape [L, max_len, KV, hd], filled left to right.

    Memory per token = 2 (K and V) * L * KV * hd * bytes.  For Llama-2-7B in
    fp16 that is 2 * 32 * 32 * 128 * 2 = 512 KiB *per token*.
    """

    def __init__(self, cfg: LlamaConfig, max_len: int | None = None):
        self.cfg = cfg
        self.max_len = max_len or cfg.max_seq_len
        shape = (cfg.n_layers, self.max_len, cfg.n_kv_heads, cfg.head_dim)
        self.k = np.zeros(shape, dtype=F32)
        self.v = np.zeros(shape, dtype=F32)
        self.length = 0

    @property
    def nbytes(self) -> int:
        return int(self.k.nbytes + self.v.nbytes)

    def truncate(self, length: int) -> None:
        """Roll back (used by speculative decoding when draft tokens are rejected)."""
        if not 0 <= length <= self.length:
            raise ValueError("invalid truncate length")
        self.length = length

    def context(self, n_tokens: int, all_logits: bool = False, kernels=None) -> _ContiguousContext:
        if n_tokens <= 0:
            raise ValueError("n_tokens must be positive")
        if self.length + n_tokens > self.max_len:
            raise ValueError(
                f"KV cache overflow: {self.length} + {n_tokens} > {self.max_len} tokens"
            )
        return _ContiguousContext(self, n_tokens, all_logits, kernels)


class _ContiguousContext:
    def __init__(self, cache: KVCache, n_tokens: int, all_logits: bool, kernels):
        self.cache = cache
        self.start = cache.length
        self.n = n_tokens
        self.positions = np.arange(self.start, self.start + n_tokens, dtype=np.int64)
        self.logit_indices = None if all_logits else np.array([n_tokens - 1])
        self._attention = kernels.attention if kernels is not None else ops.attention

    def attend(self, layer: int, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        c, s, e = self.cache, self.start, self.start + self.n
        c.k[layer, s:e] = k  # append the new keys/values ...
        c.v[layer, s:e] = v
        # ... and attend over everything cached so far (causal inside the chunk)
        return self._attention(q, c.k[layer, :e], c.v[layer, :e], q_offset=s, causal=True)

    def finish(self) -> None:
        self.cache.length = self.start + self.n


# ---------------------------------------------------------------------------
# Profiling helper: where does the time go?
# ---------------------------------------------------------------------------
@dataclass
class Profiler:
    enabled: bool = False
    seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def reset(self) -> None:
        self.seconds = defaultdict(float)

    def summary(self) -> dict[str, float]:
        return dict(sorted(self.seconds.items(), key=lambda kv: -kv[1]))


class _Timer:
    __slots__ = ("p", "name", "t0")

    def __init__(self, p: Profiler, name: str):
        self.p, self.name = p, name

    def __enter__(self):
        if self.p.enabled:
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.p.enabled:
            self.p.seconds[self.name] += time.perf_counter() - self.t0
        return False


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class LlamaModel:
    def __init__(self, cfg: LlamaConfig, weights: LlamaWeights, kernels=None):
        if len(weights.layers) != cfg.n_layers:
            raise ValueError("number of layer weights does not match config")
        self.cfg = cfg
        self.w = weights
        self.kernels = kernels  # optional C++ kernel backend (inferlab.kernels.cpp.CppKernels)
        self.profiler = Profiler()
        if kernels is not None:  # quantized layers may then use the C++ integer GEMV
            for lw in weights.layers:
                for m in (lw.wq, lw.wk, lw.wv, lw.wo, lw.w1, lw.w2, lw.w3):
                    if hasattr(m, "use_cpp"):
                        m.use_cpp = True

    # -- helpers ------------------------------------------------------------
    def _rmsnorm(self, x, w):
        if self.kernels is not None:
            return self.kernels.rmsnorm(x, w, self.cfg.norm_eps)
        return ops.rmsnorm(x, w, self.cfg.norm_eps)

    def _swiglu(self, gate, up):
        if self.kernels is not None:
            return self.kernels.swiglu(gate, up)
        return ops.swiglu(gate, up)

    def new_kv_cache(self, max_len: int | None = None) -> KVCache:
        return KVCache(self.cfg, max_len)

    @property
    def nbytes(self) -> int:
        return self.w.nbytes()

    def truncated(self, n_layers: int) -> LlamaModel:
        """A model that shares weights but only runs the first ``n_layers`` layers.

        Used as a zero-cost *draft model* for self-speculative decoding.
        """
        if not 1 <= n_layers <= self.cfg.n_layers:
            raise ValueError("n_layers out of range")
        from dataclasses import replace

        cfg = replace(self.cfg, name=f"{self.cfg.name}-first{n_layers}", n_layers=n_layers)
        w = LlamaWeights(
            self.w.tok_embeddings,
            self.w.layers[:n_layers],
            self.w.final_norm,
            self.w.lm_head,
            self.w.tied,
        )
        return LlamaModel(cfg, w, self.kernels)

    # -- forward ------------------------------------------------------------
    def forward(self, tokens: np.ndarray, ctx: ForwardContext) -> np.ndarray:
        """Run the transformer on a flat batch of tokens.

        tokens: [T] int token ids. ``ctx.positions`` gives each token's
        position. Returns logits [n, vocab] for rows ``ctx.logit_indices``.
        """
        cfg, P = self.cfg, self.profiler
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.ndim != 1 or tokens.size == 0:
            raise ValueError("tokens must be a non-empty 1-D array")
        if tokens.min() < 0 or tokens.max() >= cfg.vocab_size:
            raise ValueError("token id out of range")
        T, H, KV, hd = tokens.shape[0], cfg.n_heads, cfg.n_kv_heads, cfg.head_dim

        with _Timer(P, "embedding"):
            x = np.asarray(self.w.tok_embeddings[tokens], dtype=F32)  # [T, D]

        for li, lw in enumerate(self.w.layers):
            # ---- attention block ----
            with _Timer(P, "rmsnorm"):
                h = self._rmsnorm(x, lw.attn_norm)
            with _Timer(P, "qkv_proj"):
                q = lw.wq(h).reshape(T, H, hd)
                k = lw.wk(h).reshape(T, KV, hd)
                v = lw.wv(h).reshape(T, KV, hd)
            with _Timer(P, "rope"):
                q = ops.apply_rope(q, ctx.positions, cfg.rope_theta, cfg.rope_style)
                k = ops.apply_rope(k, ctx.positions, cfg.rope_theta, cfg.rope_style)
            with _Timer(P, "attention"):
                a = ctx.attend(li, q, k, v).reshape(T, H * hd)
            with _Timer(P, "o_proj"):
                x = x + lw.wo(a)
            # ---- feed-forward block ----
            with _Timer(P, "rmsnorm"):
                h = self._rmsnorm(x, lw.ffn_norm)
            with _Timer(P, "ffn"):
                x = x + lw.w2(self._swiglu(lw.w1(h), lw.w3(h)))

        ctx.finish()
        if ctx.logit_indices is not None:
            x = x[ctx.logit_indices]
        with _Timer(P, "rmsnorm"):
            x = self._rmsnorm(x, self.w.final_norm)
        with _Timer(P, "lm_head"):
            return np.asarray(self.w.lm_head(x), dtype=F32)

    # -- convenience --------------------------------------------------------
    def prefill(self, tokens, cache: KVCache, all_logits: bool = False) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.int64)
        return self.forward(tokens, cache.context(len(tokens), all_logits, self.kernels))

    def logits_full(self, tokens) -> np.ndarray:
        """Logits for every position of ``tokens`` without keeping a cache."""
        cache = self.new_kv_cache(len(tokens))
        return self.prefill(tokens, cache, all_logits=True)


def requantize(model: LlamaModel, kind: str) -> LlamaModel:
    """Return a copy of ``model`` whose linear layers use ``kind`` (fp16, int8, q4_0...)."""

    def conv(m):
        return make_linear(m.dequantize(), kind)

    layers = [
        LayerWeights(
            lw.attn_norm,
            conv(lw.wq),
            conv(lw.wk),
            conv(lw.wv),
            conv(lw.wo),
            lw.ffn_norm,
            conv(lw.w1),
            conv(lw.w2),
            conv(lw.w3),
        )
        for lw in model.w.layers
    ]
    w = LlamaWeights(
        model.w.tok_embeddings, layers, model.w.final_norm, model.w.lm_head, model.w.tied
    )
    return LlamaModel(model.cfg, w, model.kernels)
