import numpy as np
import pytest

from inferlab.foundations import ops
from inferlab.kernels import cpp, reference
from inferlab.kernels.bench import bench_attention, bench_gemv, bench_matmul, bench_softmax

needs_cpp = pytest.mark.skipif(not cpp.available(), reason="C++ kernels not built (make kernels)")
F32 = np.float32


# ---------------------------------------------------------------------------
# NumPy references (always run)
# ---------------------------------------------------------------------------
def test_reference_flash_attention(rng):
    q = rng.standard_normal((12, 4, 8), dtype=F32)
    k = rng.standard_normal((20, 2, 8), dtype=F32)
    v = rng.standard_normal((20, 2, 8), dtype=F32)
    for bk in (1, 5, 32):
        np.testing.assert_allclose(reference.flash_attention(q, k, v, 8, True, bk), ops.attention(q, k, v, 8, True),
                                   atol=1e-5)


def test_reference_online_softmax_and_tiled_matmul(rng):
    x = rng.standard_normal(100) * 10
    np.testing.assert_allclose(reference.online_softmax(x), ops.softmax(x), atol=1e-12)
    a, b = rng.standard_normal((40, 50), dtype=F32), rng.standard_normal((50, 30), dtype=F32)
    np.testing.assert_allclose(reference.tiled_matmul(a, b, 16), a @ b, atol=1e-4)


def test_reference_paged_attention(rng):
    bs, nb, KV, H, hd, ctx = 4, 8, 2, 4, 8, 13
    k = rng.standard_normal((ctx, KV, hd), dtype=F32)
    v = rng.standard_normal((ctx, KV, hd), dtype=F32)
    table = np.array([6, 1, 3, 7])
    kc = np.zeros((nb, bs, KV, hd), F32)
    vc = np.zeros((nb, bs, KV, hd), F32)
    for p in range(ctx):
        kc[table[p // bs], p % bs], vc[table[p // bs], p % bs] = k[p], v[p]
    q = rng.standard_normal((H, hd), dtype=F32)
    ref = ops.attention(q[None], k, v, ctx - 1, True)[0]
    np.testing.assert_allclose(reference.paged_attention_decode(q, kc, vc, table, ctx), ref, atol=1e-5)
    if cpp.available():
        np.testing.assert_allclose(cpp.paged_attention_decode(q, kc, vc, table, ctx), ref, atol=1e-5)


# ---------------------------------------------------------------------------
# C++ kernels
# ---------------------------------------------------------------------------
@needs_cpp
@pytest.mark.parametrize("variant", cpp.MATMUL_VARIANTS)
def test_cpp_matmul(rng, variant):
    a = rng.standard_normal((33, 70), dtype=F32)
    b = rng.standard_normal((70, 45), dtype=F32)
    np.testing.assert_allclose(cpp.matmul(a, b, variant), a @ b, atol=1e-3)


@needs_cpp
def test_cpp_gemv_and_quantized(rng):
    W = rng.standard_normal((64, 256), dtype=F32) * 0.05
    x = rng.standard_normal(256, dtype=F32)
    np.testing.assert_allclose(cpp.gemv(W, x), W @ x, atol=1e-4)
    for kind, tol in (("q8_0", 0.02), ("q4_0", 0.2)):
        qm = cpp.QuantizedMatrix(W, kind)
        rel = np.linalg.norm(qm.gemv(x) - W @ x) / np.linalg.norm(W @ x)
        assert rel < tol
    assert cpp.QuantizedMatrix(W, "q4_0").nbytes < cpp.QuantizedMatrix(W, "q8_0").nbytes < W.nbytes


@needs_cpp
def test_cpp_elementwise(rng):
    x = rng.standard_normal((5, 64), dtype=F32)
    w = rng.standard_normal(64, dtype=F32)
    np.testing.assert_allclose(cpp.rmsnorm(x, w), ops.rmsnorm(x, w), atol=1e-5)
    for online in (True, False):
        np.testing.assert_allclose(cpp.softmax(x * 5, online), ops.softmax(x * 5), atol=1e-6)
    np.testing.assert_allclose(cpp.silu_mul(x, x * 2), ops.swiglu(x, x * 2), atol=1e-5)
    q = rng.standard_normal((4, 3, 16), dtype=F32)
    pos = np.array([0, 3, 9, 100])
    for interleaved, style in ((True, "interleaved"), (False, "half")):
        np.testing.assert_allclose(cpp.rope(q, pos, interleaved=interleaved), ops.apply_rope(q, pos, style=style),
                                   atol=1e-4)


@needs_cpp
@pytest.mark.parametrize("causal", [True, False])
def test_cpp_attention(rng, causal):
    q = rng.standard_normal((17, 4, 32), dtype=F32)
    k = rng.standard_normal((40, 2, 32), dtype=F32)
    v = rng.standard_normal((40, 2, 32), dtype=F32)
    ref = ops.attention(q, k, v, 23, causal)
    np.testing.assert_allclose(cpp.attention(q, k, v, 23, causal, flash=False), ref, atol=1e-5)
    for bq, bk in ((1, 1), (4, 7), (16, 64)):
        np.testing.assert_allclose(cpp.attention(q, k, v, 23, causal, True, bq, bk), ref, atol=1e-5)


@needs_cpp
def test_cpp_paged_attention_rejects_bad_block_ids(rng):
    kc = np.zeros((4, 2, 1, 8), F32)
    q = np.zeros((1, 8), F32)
    with pytest.raises(cpp.KernelError, match="out of bounds"):
        cpp.paged_attention_decode(q, kc, kc, np.array([0, 99]), 3)
    with pytest.raises(cpp.KernelError):
        cpp.paged_attention_decode(q, kc, kc, np.array([0]), 3)  # table too short for context


@needs_cpp
def test_cpp_input_validation():
    with pytest.raises(ValueError):
        cpp.matmul(np.zeros((2, 3), F32), np.zeros((4, 2), F32))
    with pytest.raises(ValueError):
        cpp.gemv(np.zeros((2, 3), F32), np.zeros(4, F32))
    with pytest.raises(ValueError):
        cpp.QuantizedMatrix(np.zeros((2, 30), F32))


@needs_cpp
def test_model_with_cpp_kernels_matches_numpy(tiny_model):
    from inferlab.foundations.model import LlamaModel

    m = LlamaModel(tiny_model.cfg, tiny_model.w, cpp.CppKernels())
    toks = [1, 20, 30, 40, 50]
    np.testing.assert_allclose(m.logits_full(toks), tiny_model.logits_full(toks), atol=1e-3)


def test_benchmarks_run_small():
    assert bench_matmul(32)["results"]
    assert bench_gemv(64, 64)["results"]
    assert bench_attention(32, 2, 32)["results"]
    assert bench_softmax(2, 64)["results"]
    with pytest.raises(ValueError):
        bench_matmul(100000)


@needs_cpp
@pytest.mark.parametrize("kind", ["q8_0", "q4_0"])
def test_quantized_model_uses_cpp_integer_gemv(tiny_model, kind):
    from inferlab.foundations.model import LlamaModel, requantize

    q_np = requantize(tiny_model, kind)
    q_cpp = LlamaModel(q_np.cfg, q_np.w, cpp.CppKernels())
    assert all(m.use_cpp for lw in q_cpp.w.layers for m in (lw.wq, lw.w2))
    cache_np, cache_cpp = q_np.new_kv_cache(8), q_cpp.new_kv_cache(8)
    toks = np.array([1, 30, 40])
    q_np.prefill(toks, cache_np)
    q_cpp.prefill(toks, cache_cpp)
    a = q_np.forward(np.array([50]), cache_np.context(1))  # decode step: cpp path = integer GEMV
    b = q_cpp.forward(np.array([50]), cache_cpp.context(1))
    assert np.argmax(a) == np.argmax(b)
    assert np.linalg.norm(a - b) / np.linalg.norm(a) < 0.05  # activations are quantized to Q8_0 too
    for lw in q_np.w.layers:  # restore: the weights objects are shared with q_np
        for m in (lw.wq, lw.wk, lw.wv, lw.wo, lw.w1, lw.w2, lw.w3):
            m.use_cpp = False
