"""ctypes bindings for the C++ kernels in ``csrc/``.

Build the library first:  ``make kernels``  (CMake -> inferlab/kernels/lib/).

Safety rules followed here (FFI is where memory bugs hide):

* the shared library is loaded **only** from this package's ``lib/``
  directory - never from the current directory, ``LD_LIBRARY_PATH`` or a
  user-supplied path (prevents library hijacking);
* every array is converted to a C-contiguous array of the exact dtype and its
  shape is checked in Python *before* a pointer crosses into C;
* the C side re-validates sizes and bounds and returns a status code, which
  we always check.

ctypes releases the GIL during the call, so kernels run in parallel with the
asyncio event loop of the server.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

F32 = np.float32
_LIB_DIR = Path(__file__).resolve().parent / "lib"
_NAMES = {"linux": "libinferlab_kernels.so", "darwin": "libinferlab_kernels.dylib", "win32": "inferlab_kernels.dll"}

_P_F32 = ctypes.POINTER(ctypes.c_float)
_P_I8 = ctypes.POINTER(ctypes.c_int8)
_P_U8 = ctypes.POINTER(ctypes.c_uint8)
_P_I32 = ctypes.POINTER(ctypes.c_int32)
_P_I64 = ctypes.POINTER(ctypes.c_int64)
_I64 = ctypes.c_int64

_ERRORS = {
    -1: "null pointer",
    -2: "invalid size",
    -3: "invalid argument",
    -4: "index out of bounds",
    -5: "out of memory",
}

_lib = None
_load_error: str | None = None


class KernelError(RuntimeError):
    pass


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    name = _NAMES.get(sys.platform, "libinferlab_kernels.so")
    path = _LIB_DIR / name
    if not path.is_file():
        _load_error = f"{path} not found - build it with `make kernels`"
        return None
    # OpenMP threads spin-wait by default; next to NumPy's OpenBLAS threads that
    # oversubscribes the cores (2 pools fighting = 4x slower decode on a 4-core
    # machine). Passive waiting must be set before libgomp initialises.
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:
        _load_error = f"failed to load {path.name}: {exc}"
        return None
    sig = {
        "il_matmul_naive": [_P_F32, _P_F32, _P_F32, _I64, _I64, _I64],
        "il_matmul_reordered": [_P_F32, _P_F32, _P_F32, _I64, _I64, _I64],
        "il_matmul_tiled": [_P_F32, _P_F32, _P_F32, _I64, _I64, _I64],
        "il_matmul_parallel": [_P_F32, _P_F32, _P_F32, _I64, _I64, _I64],
        "il_gemv": [_P_F32, _P_F32, _P_F32, _I64, _I64],
        "il_quantize_q8_0": [_P_F32, _I64, _P_I8, _P_F32],
        "il_quantize_q4_0": [_P_F32, _I64, _P_U8, _P_F32],
        "il_gemv_q8_0": [_P_I8, _P_F32, _P_F32, _P_F32, _I64, _I64],
        "il_gemv_q4_0": [_P_U8, _P_F32, _P_F32, _P_F32, _I64, _I64],
        "il_rmsnorm": [_P_F32, _P_F32, _P_F32, _I64, _I64, ctypes.c_float],
        "il_fused_add_rmsnorm": [_P_F32, _P_F32, _P_F32, _I64, _I64, ctypes.c_float],
        "il_softmax_naive": [_P_F32, _I64, _I64],
        "il_softmax_online": [_P_F32, _I64, _I64],
        "il_silu_mul": [_P_F32, _P_F32, _P_F32, _I64],
        "il_rope": [_P_F32, _I64, _I64, _I64, _P_I64, ctypes.c_float, ctypes.c_int32],
        "il_attention_naive": [_P_F32] * 4 + [_I64] * 6 + [ctypes.c_int32],
        "il_flash_attention": [_P_F32] * 4 + [_I64] * 6 + [ctypes.c_int32, _I64, _I64],
        "il_paged_attention_decode": [_P_F32, _P_F32, _P_F32, _P_I32] + [_I64] * 7 + [_P_F32],
    }
    for fname, argtypes in sig.items():
        fn = getattr(lib, fname)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int32
    for fname in ("il_version", "il_simd_name"):
        getattr(lib, fname).restype = ctypes.c_char_p
        getattr(lib, fname).argtypes = []
    lib.il_has_openmp.restype = ctypes.c_int32
    lib.il_get_max_threads.restype = ctypes.c_int32
    lib.il_set_num_threads.argtypes = [ctypes.c_int32]
    lib.il_set_num_threads.restype = None
    _lib = lib
    return lib


def available() -> bool:
    return _load() is not None


def load_error() -> str | None:
    _load()
    return _load_error


def info() -> dict:
    lib = _load()
    if lib is None:
        return {"available": False, "error": _load_error}
    return {
        "available": True,
        "version": lib.il_version().decode(),
        "simd": lib.il_simd_name().decode(),
        "openmp": bool(lib.il_has_openmp()),
        "threads": int(lib.il_get_max_threads()),
    }


def _lib_or_raise():
    lib = _load()
    if lib is None:
        raise KernelError(_load_error or "kernel library unavailable")
    return lib


def _check(status: int, name: str) -> None:
    if status != 0:
        raise KernelError(f"{name} failed: {_ERRORS.get(status, status)}")


def _arr(a, dtype, shape=None) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=dtype)
    if shape is not None and a.shape != tuple(shape):
        raise ValueError(f"expected shape {tuple(shape)}, got {a.shape}")
    return a


def _p(a: np.ndarray, ptype):
    return a.ctypes.data_as(ptype)


# ---------------------------------------------------------------------------
# GEMM / GEMV
# ---------------------------------------------------------------------------
MATMUL_VARIANTS = ("naive", "reordered", "tiled", "parallel")


def matmul(A, B, variant: str = "parallel") -> np.ndarray:
    if variant not in MATMUL_VARIANTS:
        raise ValueError(f"variant must be one of {MATMUL_VARIANTS}")
    lib = _lib_or_raise()
    A = _arr(A, F32)
    B = _arr(B, F32)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError("matmul expects A[M,K] and B[K,N]")
    M, K = A.shape
    N = B.shape[1]
    C = np.empty((M, N), dtype=F32)
    fn = getattr(lib, f"il_matmul_{variant}")
    _check(fn(_p(A, _P_F32), _p(B, _P_F32), _p(C, _P_F32), M, N, K), fn.__name__)
    return C


def gemv(W, x) -> np.ndarray:
    lib = _lib_or_raise()
    W = _arr(W, F32)
    if W.ndim != 2:
        raise ValueError("W must be 2-D")
    N, K = W.shape
    x = _arr(x, F32, (K,))
    y = np.empty(N, dtype=F32)
    _check(lib.il_gemv(_p(W, _P_F32), _p(x, _P_F32), _p(y, _P_F32), N, K), "il_gemv")
    return y


def quantized_gemv(kind: str, q, d, x) -> np.ndarray:
    """y = W x for llama.cpp-layout Q8_0 (q int8 [N,K]) or Q4_0 (q uint8 [N,K/2]) weights, d float32 [N,K/32]."""
    lib = _lib_or_raise()
    if q.ndim != 2:
        raise ValueError("q must be 2-D")
    N = q.shape[0]
    K = q.shape[1] * (2 if kind == "q4_0" else 1)
    if K % 32:
        raise ValueError("K must be a multiple of 32")
    d = _arr(d, F32, (N, K // 32))
    x = _arr(x, F32, (K,))
    y = np.empty(N, dtype=F32)
    if kind == "q8_0":
        qa = _arr(q, np.int8, (N, K))
        st = lib.il_gemv_q8_0(_p(qa, _P_I8), _p(d, _P_F32), _p(x, _P_F32), _p(y, _P_F32), N, K)
    elif kind == "q4_0":
        qa = _arr(q, np.uint8, (N, K // 2))
        st = lib.il_gemv_q4_0(_p(qa, _P_U8), _p(d, _P_F32), _p(x, _P_F32), _p(y, _P_F32), N, K)
    else:
        raise ValueError("kind must be q8_0 or q4_0")
    _check(st, f"gemv_{kind}")
    return y


class QuantizedMatrix:
    """W quantized by the C++ kernels (q8_0 or q4_0, float32 block scales)."""

    def __init__(self, W, kind: str = "q8_0"):
        if kind not in ("q8_0", "q4_0"):
            raise ValueError("kind must be q8_0 or q4_0")
        lib = _lib_or_raise()
        W = _arr(W, F32)
        if W.ndim != 2 or W.shape[1] % 32 != 0:
            raise ValueError("W must be 2-D with in_features divisible by 32")
        self.kind = kind
        self.N, self.K = W.shape
        n = W.size
        self.d = np.empty(n // 32, dtype=F32)
        if kind == "q8_0":
            self.q = np.empty(n, dtype=np.int8)
            _check(lib.il_quantize_q8_0(_p(W, _P_F32), n, _p(self.q, _P_I8), _p(self.d, _P_F32)), "quantize_q8_0")
        else:
            self.q = np.empty(n // 2, dtype=np.uint8)
            _check(lib.il_quantize_q4_0(_p(W, _P_F32), n, _p(self.q, _P_U8), _p(self.d, _P_F32)), "quantize_q4_0")

    @property
    def nbytes(self) -> int:
        return int(self.q.nbytes + self.d.nbytes)

    def gemv(self, x) -> np.ndarray:
        lib = _lib_or_raise()
        x = _arr(x, F32, (self.K,))
        y = np.empty(self.N, dtype=F32)
        if self.kind == "q8_0":
            st = lib.il_gemv_q8_0(_p(self.q, _P_I8), _p(self.d, _P_F32), _p(x, _P_F32), _p(y, _P_F32), self.N, self.K)
        else:
            st = lib.il_gemv_q4_0(_p(self.q, _P_U8), _p(self.d, _P_F32), _p(x, _P_F32), _p(y, _P_F32), self.N, self.K)
        _check(st, f"gemv_{self.kind}")
        return y


# ---------------------------------------------------------------------------
# elementwise
# ---------------------------------------------------------------------------
def rmsnorm(x, w, eps: float = 1e-5) -> np.ndarray:
    lib = _lib_or_raise()
    x = _arr(x, F32)
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    w = _arr(w, F32, (shape[-1],))
    y = np.empty_like(x2)
    _check(lib.il_rmsnorm(_p(x2, _P_F32), _p(w, _P_F32), _p(y, _P_F32), x2.shape[0], x2.shape[1], eps), "rmsnorm")
    return y.reshape(shape)


def softmax(x, online: bool = True) -> np.ndarray:
    lib = _lib_or_raise()
    y = np.array(x, dtype=F32, copy=True, order="C")
    y2 = y.reshape(-1, y.shape[-1])
    fn = lib.il_softmax_online if online else lib.il_softmax_naive
    _check(fn(_p(y2, _P_F32), y2.shape[0], y2.shape[1]), "softmax")
    return y


def silu_mul(gate, up) -> np.ndarray:
    lib = _lib_or_raise()
    gate = _arr(gate, F32)
    up = _arr(up, F32, gate.shape)
    out = np.empty_like(gate)
    _check(lib.il_silu_mul(_p(gate, _P_F32), _p(up, _P_F32), _p(out, _P_F32), gate.size), "silu_mul")
    return out


def rope(x, positions, theta: float = 10000.0, interleaved: bool = True) -> np.ndarray:
    lib = _lib_or_raise()
    y = np.array(x, dtype=F32, copy=True, order="C")
    if y.ndim != 3:
        raise ValueError("rope expects [T, heads, head_dim]")
    T, H, hd = y.shape
    pos = _arr(positions, np.int64, (T,))
    _check(lib.il_rope(_p(y, _P_F32), T, H, hd, _p(pos, _P_I64), theta, int(interleaved)), "rope")
    return y


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------
def _attn_args(q, k, v):
    q = _arr(q, F32)
    k = _arr(k, F32)
    v = _arr(v, F32, k.shape)
    if q.ndim != 3 or k.ndim != 3 or q.shape[2] != k.shape[2]:
        raise ValueError("attention expects q[T,H,hd], k/v[S,KV,hd]")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("n_heads must be a multiple of n_kv_heads")
    return q, k, v


def attention(q, k, v, q_offset: int = 0, causal: bool = True, flash: bool = True,
              block_q: int = 16, block_k: int = 64) -> np.ndarray:
    lib = _lib_or_raise()
    q, k, v = _attn_args(q, k, v)
    Tq, H, hd = q.shape
    Tk, KV, _ = k.shape
    o = np.empty_like(q)
    args = (_p(q, _P_F32), _p(k, _P_F32), _p(v, _P_F32), _p(o, _P_F32), Tq, Tk, H, KV, hd, int(q_offset), int(causal))
    if flash:
        _check(lib.il_flash_attention(*args, block_q, block_k), "flash_attention")
    else:
        _check(lib.il_attention_naive(*args), "attention_naive")
    return o


def paged_attention_decode(q, k_cache, v_cache, block_table, context_len: int) -> np.ndarray:
    """q[H, hd]; k/v_cache[num_blocks, block_size, KV, hd]; returns out[H, hd]."""
    lib = _lib_or_raise()
    k_cache = _arr(k_cache, F32)
    v_cache = _arr(v_cache, F32, k_cache.shape)
    if k_cache.ndim != 4:
        raise ValueError("cache must be [num_blocks, block_size, KV, hd]")
    nb, bs, KV, hd = k_cache.shape
    q = _arr(q, F32)
    if q.ndim != 2 or q.shape[1] != hd:
        raise ValueError("q must be [H, hd]")
    table = _arr(block_table, np.int32)
    out = np.empty_like(q)
    _check(
        lib.il_paged_attention_decode(
            _p(q, _P_F32), _p(k_cache, _P_F32), _p(v_cache, _P_F32), _p(table, _P_I32),
            table.size, int(context_len), nb, bs, q.shape[0], KV, hd, _p(out, _P_F32),
        ),
        "paged_attention_decode",
    )
    return out


class CppKernels:
    """Kernel backend for ``LlamaModel(kernels=...)`` - swaps NumPy ops for C++."""

    name = "cpp"

    def __init__(self):
        _lib_or_raise()

    def rmsnorm(self, x, w, eps):
        return rmsnorm(x, w, eps)

    def swiglu(self, gate, up):
        return silu_mul(gate, up)

    def attention(self, q, k, v, q_offset=0, causal=True):
        return attention(q, k, v, q_offset, causal, flash=True)

    def paged_attention_decode(self, q, k_cache, v_cache, block_table, context_len):
        return paged_attention_decode(q, k_cache, v_cache, block_table, context_len)
