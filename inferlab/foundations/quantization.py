"""Weight quantization - shrinking the bytes that every decode step must read.

Decode is memory-bandwidth bound (see ``perfmodel.py``): each generated token
reads *all* weights once. Halving bytes per weight roughly doubles decode
speed. This module implements the formats you will meet in practice:

* ``fp16`` / ``bf16`` - 2 bytes, used by vLLM / SGLang / TensorRT-LLM by default
* ``int8`` per-output-channel symmetric (weight-only, like LLM.int8 / W8A16)
* ``q8_0`` and ``q4_0`` - llama.cpp's GGUF block formats: blocks of 32 values
  share one scale. Q4_0 costs 4.5 bits/weight (4 bits + one fp16 scale / 32).

Linear layers expose ``__call__(x) -> x @ W.T`` so the model does not care
which format a layer uses.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

F32 = np.float32
QK = 32  # llama.cpp block size for q8_0 / q4_0


class LinearLike(Protocol):
    in_features: int
    out_features: int
    kind: str

    def __call__(self, x: np.ndarray) -> np.ndarray: ...

    @property
    def nbytes(self) -> int: ...

    def dequantize(self) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Full precision / half precision
# ---------------------------------------------------------------------------
class Linear:
    """y = x @ W.T with W stored as [out_features, in_features] (PyTorch layout)."""

    kind = "fp32"

    def __init__(self, weight: np.ndarray):
        if weight.ndim != 2:
            raise ValueError("Linear weight must be 2-D")
        self.weight = weight if weight.dtype == F32 else weight.astype(F32)
        self.out_features, self.in_features = self.weight.shape

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight.T

    @property
    def nbytes(self) -> int:
        return int(self.weight.size * 4)

    def dequantize(self) -> np.ndarray:
        return np.asarray(self.weight, dtype=F32)


def bf16_round(x: np.ndarray) -> np.ndarray:
    """Round float32 -> bfloat16 (round-to-nearest-even) and back to float32.

    bf16 keeps fp32's 8 exponent bits but only 7 mantissa bits, so it never
    overflows where fp32 would not - the reason training and serving moved to it.
    """
    u = np.ascontiguousarray(x, dtype=F32).view(np.uint32)
    rounding_bias = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + rounding_bias) & np.uint32(0xFFFF0000)).view(F32)


class HalfLinear:
    """Weights stored in fp16 (or bf16-rounded fp32), upcast to fp32 for the matmul."""

    def __init__(self, weight: np.ndarray, kind: str = "fp16"):
        if kind not in ("fp16", "bf16"):
            raise ValueError("kind must be fp16 or bf16")
        self.kind = kind
        w = np.asarray(weight, dtype=F32)
        if kind == "fp16":
            self.weight = w.astype(np.float16)
        else:  # store the upper 16 bits of the bf16-rounded value
            self.weight = (bf16_round(w).view(np.uint32) >> 16).astype(np.uint16)
        self.out_features, self.in_features = w.shape

    def dequantize(self) -> np.ndarray:
        if self.kind == "fp16":
            return self.weight.astype(F32)
        return (self.weight.astype(np.uint32) << 16).view(F32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.dequantize().T

    @property
    def nbytes(self) -> int:
        return int(self.weight.size * 2)


# ---------------------------------------------------------------------------
# INT8 per-channel (weight-only)
# ---------------------------------------------------------------------------
def quantize_int8_per_channel(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-output-channel: scale_r = max|W_r| / 127, q = round(W / scale)."""
    w = np.asarray(w, dtype=F32)
    amax = np.max(np.abs(w), axis=1, keepdims=True)
    scale = np.where(amax == 0, 1.0, amax / 127.0).astype(F32)
    q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    return q, scale[:, 0]


class Int8Linear:
    kind = "int8"

    def __init__(self, weight: np.ndarray):
        self.q, self.scale = quantize_int8_per_channel(weight)
        self.out_features, self.in_features = self.q.shape

    def dequantize(self) -> np.ndarray:
        return self.q.astype(F32) * self.scale[:, None]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # (x @ q.T) * scale == x @ (q * scale).T  -> apply scale after the matmul
        return (x @ self.q.T.astype(F32)) * self.scale[None, :]

    @property
    def nbytes(self) -> int:
        return int(self.q.size + self.scale.size * 4)


# ---------------------------------------------------------------------------
# llama.cpp block formats: Q8_0 and Q4_0
# ---------------------------------------------------------------------------
def _blocks(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=F32)
    if w.shape[-1] % QK != 0:
        raise ValueError(f"last dimension must be a multiple of {QK}")
    return w.reshape(*w.shape[:-1], w.shape[-1] // QK, QK)


def quantize_q8_0(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per block of 32: d = max|x| / 127, q = round(x / d). Returns (q int8, d fp16)."""
    b = _blocks(w)
    amax = np.max(np.abs(b), axis=-1, keepdims=True)
    d = (amax / 127.0).astype(F32)
    inv = np.where(d == 0, 0.0, 1.0 / np.where(d == 0, 1.0, d))
    q = np.clip(np.rint(b * inv), -127, 127).astype(np.int8)
    return q.reshape(w.shape), d[..., 0].astype(np.float16)


def dequantize_q8_0(q: np.ndarray, d: np.ndarray) -> np.ndarray:
    b = q.reshape(*q.shape[:-1], q.shape[-1] // QK, QK).astype(F32)
    return (b * d.astype(F32)[..., None]).reshape(q.shape)


def quantize_q4_0(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """llama.cpp Q4_0: d = max / -8 (max = signed value with largest |x|),
    q = clamp(round(x / d + 8), 0, 15), stored as packed nibbles.

    Returns (packed uint8 [..., K/2], d fp16 [..., K/32]).
    Packing follows ggml: byte j of a block holds element j (low nibble)
    and element j + 16 (high nibble).
    """
    b = _blocks(w)
    idx = np.argmax(np.abs(b), axis=-1)
    mx = np.take_along_axis(b, idx[..., None], axis=-1)
    d = (mx / -8.0).astype(F32)
    inv = np.where(d == 0, 0.0, 1.0 / np.where(d == 0, 1.0, d))
    q = np.clip(np.floor(b * inv + 8.5), 0, 15).astype(np.uint8)
    lo, hi = q[..., :16], q[..., 16:]
    packed = (lo | (hi << 4)).astype(np.uint8)
    out_shape = (*w.shape[:-1], w.shape[-1] // 2)
    return packed.reshape(out_shape), d[..., 0].astype(np.float16)


def dequantize_q4_0(packed: np.ndarray, d: np.ndarray) -> np.ndarray:
    K = packed.shape[-1] * 2
    p = packed.reshape(*packed.shape[:-1], K // QK, QK // 2)
    lo = (p & 0x0F).astype(np.int8) - 8
    hi = (p >> 4).astype(np.int8) - 8
    vals = np.concatenate([lo, hi], axis=-1).astype(F32)
    return (vals * d.astype(F32)[..., None]).reshape(*packed.shape[:-1], K)


def _cpp_gemv_rows(lin, x: np.ndarray) -> np.ndarray:
    """Decode path: integer GEMV in C++ on the packed weights (no dequantized copy)."""
    from inferlab.kernels import cpp  # lazy: foundations must not require the C++ library

    if lin._d32 is None:
        lin._d32 = np.ascontiguousarray(lin.d, dtype=F32)
    return np.stack([cpp.quantized_gemv(lin.kind, lin.q, lin._d32, row) for row in x])


class _BlockQuantLinear:
    """Shared logic of Q8_0 / Q4_0 layers.

    ``use_cpp`` is switched on by ``LlamaModel`` when a C++ kernel backend is
    active: for small batches (decode) the packed weights then go straight to
    the AVX2 integer GEMV instead of being dequantized to fp32 on every call.
    """

    use_cpp = False
    _d32 = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.use_cpp and x.ndim == 2 and x.shape[0] <= 4:
            return _cpp_gemv_rows(self, x)
        return x @ self.dequantize().T


class Q8_0Linear(_BlockQuantLinear):
    kind = "q8_0"

    def __init__(self, weight: np.ndarray):
        self.q, self.d = quantize_q8_0(weight)
        self.out_features, self.in_features = self.q.shape

    def dequantize(self) -> np.ndarray:
        return dequantize_q8_0(self.q, self.d)

    @property
    def nbytes(self) -> int:  # 32 int8 + 1 fp16 per block = 34 bytes / 32 weights
        return int(self.q.size + self.d.size * 2)


class Q4_0Linear(_BlockQuantLinear):
    kind = "q4_0"

    def __init__(self, weight: np.ndarray):
        self.q, self.d = quantize_q4_0(weight)
        self.out_features = self.q.shape[0]
        self.in_features = self.q.shape[1] * 2

    def dequantize(self) -> np.ndarray:
        return dequantize_q4_0(self.q, self.d)

    @property
    def nbytes(self) -> int:  # 16 bytes of nibbles + 1 fp16 per block = 18 bytes / 32 weights
        return int(self.q.size + self.d.size * 2)


LINEAR_KINDS = ("fp32", "fp16", "bf16", "int8", "q8_0", "q4_0")


def make_linear(weight: np.ndarray, kind: str = "fp32") -> LinearLike:
    if kind == "fp32":
        return Linear(weight)
    if kind in ("fp16", "bf16"):
        return HalfLinear(weight, kind)
    if kind == "int8":
        return Int8Linear(weight)
    if kind == "q8_0":
        return Q8_0Linear(weight)
    if kind == "q4_0":
        return Q4_0Linear(weight)
    raise ValueError(f"unknown linear kind {kind!r}; choose from {LINEAR_KINDS}")


def error_stats(reference: np.ndarray, approx: np.ndarray) -> dict[str, float]:
    """Max abs error, RMSE and signal-to-quantization-noise ratio (dB)."""
    ref = np.asarray(reference, dtype=np.float64)
    err = ref - np.asarray(approx, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(err**2)))
    signal = float(np.sqrt(np.mean(ref**2)))
    snr = float("inf") if rmse == 0 else 20.0 * np.log10(max(signal, 1e-30) / rmse)
    return {"max_abs_error": float(np.max(np.abs(err))), "rmse": rmse, "snr_db": snr}
