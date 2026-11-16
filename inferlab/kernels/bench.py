"""Kernel micro-benchmarks used by the UI (Kernels page) and Lab 09.

All sizes are bounded so the HTTP endpoint cannot be used to burn the CPU.
"""

from __future__ import annotations

import time

import numpy as np

from inferlab.foundations import ops
from inferlab.foundations.quantization import dequantize_q4_0, quantize_q4_0
from inferlab.kernels import cpp
from inferlab.kernels.reference import flash_attention as np_flash

F32 = np.float32
MAX_GEMM = 1024
MAX_GEMV = 8192
MAX_SEQ = 2048


def _time(fn, reps: int = 3, budget_s: float = 2.0) -> float:
    # Let the previous implementation's thread pool (OpenBLAS or OpenMP) stop
    # spin-waiting first - two busy pools on the same cores = oversubscription,
    # a real production pitfall (see docs/03-kernels.md).
    time.sleep(0.15)
    fn()  # warm-up (page faults, caches, OpenMP thread pool)
    times = []
    t_end = time.perf_counter() + budget_s
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
        if time.perf_counter() > t_end:
            break
    return float(min(times))


def bench_matmul(n: int = 256, include_naive: bool = True) -> dict:
    if not 16 <= n <= MAX_GEMM:
        raise ValueError(f"n must be in [16, {MAX_GEMM}]")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n), dtype=F32)
    B = rng.standard_normal((n, n), dtype=F32)
    ref = A @ B
    flops = 2.0 * n**3
    rows = []

    def add(name, fn, reps=3):
        t = _time(fn, reps)
        out = fn()
        rows.append({"name": name, "ms": t * 1e3, "gflops": flops / t / 1e9,
                     "max_err": float(np.max(np.abs(out - ref)))})

    if cpp.available():
        if include_naive and n <= 512:
            add("C++ naive (i-j-k)", lambda: cpp.matmul(A, B, "naive"), 1)
        add("C++ loop-reordered (i-k-j)", lambda: cpp.matmul(A, B, "reordered"))
        add("C++ tiled + register-blocked", lambda: cpp.matmul(A, B, "tiled"))
        add("C++ tiled + OpenMP", lambda: cpp.matmul(A, B, "parallel"))
    add("NumPy (OpenBLAS / Accelerate)", lambda: A @ B)
    return {"op": "gemm", "shape": [n, n, n], "results": rows, "cpp": cpp.info()}


def bench_gemv(n: int = 4096, k: int = 4096) -> dict:
    """Decode-style y = W x for fp32 vs Q8_0 vs Q4_0 weights: bandwidth wins."""
    if not (32 <= n <= MAX_GEMV and 32 <= k <= MAX_GEMV and k % 32 == 0):
        raise ValueError(f"n, k must be in [32, {MAX_GEMV}] and k divisible by 32")
    rng = np.random.default_rng(1)
    W = rng.standard_normal((n, k), dtype=F32) * 0.02
    x = rng.standard_normal(k, dtype=F32)
    ref = W @ x
    rows = []

    def add(name, fn, nbytes):
        t = _time(fn, 10)
        out = fn()
        rel = float(np.linalg.norm(out - ref) / max(np.linalg.norm(ref), 1e-12))
        rows.append({"name": name, "ms": t * 1e3, "weight_mb": nbytes / 2**20,
                     "gbps": nbytes / t / 1e9, "rel_err": rel})

    add("NumPy fp32", lambda: W @ x, W.nbytes)
    if cpp.available():
        add("C++ fp32 (AVX2 FMA)", lambda: cpp.gemv(W, x), W.nbytes)
        q8 = cpp.QuantizedMatrix(W, "q8_0")
        add("C++ Q8_0 (int8 dot)", lambda: q8.gemv(x), q8.nbytes)
        q4 = cpp.QuantizedMatrix(W, "q4_0")
        add("C++ Q4_0 (4.5 bit)", lambda: q4.gemv(x), q4.nbytes)
    else:
        qq, dd = quantize_q4_0(W)
        add("NumPy Q4_0 dequant+matmul", lambda: dequantize_q4_0(qq, dd) @ x, qq.nbytes + dd.nbytes)
    return {"op": "gemv", "shape": [n, k], "results": rows, "cpp": cpp.info()}


def bench_attention(seq: int = 512, heads: int = 8, head_dim: int = 64) -> dict:
    """Prefill attention: naive (materialises T x T scores) vs FlashAttention tiling."""
    if not (16 <= seq <= MAX_SEQ and 1 <= heads <= 32 and head_dim in (32, 64, 128)):
        raise ValueError(f"seq in [16, {MAX_SEQ}], heads in [1, 32], head_dim in 32/64/128")
    rng = np.random.default_rng(2)
    q = rng.standard_normal((seq, heads, head_dim), dtype=F32)
    k = rng.standard_normal((seq, heads, head_dim), dtype=F32)
    v = rng.standard_normal((seq, heads, head_dim), dtype=F32)
    ref = ops.attention(q, k, v, 0, True)
    rows = []

    def add(name, fn, extra_mem, reps=3):
        t = _time(fn, reps)
        out = fn()
        rows.append({"name": name, "ms": t * 1e3, "scratch_mb": extra_mem / 2**20,
                     "max_err": float(np.max(np.abs(out - ref)))})

    score_bytes = heads * seq * seq * 4
    add("NumPy naive (full score matrix)", lambda: ops.attention(q, k, v, 0, True), score_bytes * 2)
    if cpp.available():
        add("C++ naive", lambda: cpp.attention(q, k, v, 0, True, flash=False), seq * seq * 4)
        add("C++ FlashAttention (tiled, online softmax)",
            lambda: cpp.attention(q, k, v, 0, True, flash=True), (16 * head_dim + 64) * 4 * cpp.info()["threads"])
    if seq <= 256:
        add("NumPy FlashAttention reference (loops)", lambda: np_flash(q, k, v, 0, True), 0, reps=1)
    return {"op": "attention", "shape": [seq, heads, head_dim], "results": rows, "cpp": cpp.info()}


def bench_softmax(rows: int = 64, cols: int = 32000) -> dict:
    if not (1 <= rows <= 256 and 16 <= cols <= 131072):
        raise ValueError("rows in [1, 256], cols in [16, 131072]")
    x = np.random.default_rng(3).standard_normal((rows, cols), dtype=F32) * 5
    ref = ops.softmax(x)
    out = []
    for name, fn in [("NumPy (3 passes)", lambda: ops.softmax(x))] + (
        [("C++ naive (3 passes)", lambda: cpp.softmax(x, online=False)),
         ("C++ online (2 passes)", lambda: cpp.softmax(x, online=True))] if cpp.available() else []
    ):
        t = _time(fn, 5)
        out.append({"name": name, "ms": t * 1e3, "max_err": float(np.max(np.abs(fn() - ref)))})
    return {"op": "softmax", "shape": [rows, cols], "results": out, "cpp": cpp.info()}


BENCHMARKS = {"matmul": bench_matmul, "gemv": bench_gemv, "attention": bench_attention, "softmax": bench_softmax}
