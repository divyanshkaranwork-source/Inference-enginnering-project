# Layer 3 — Kernels: making each operation fast

> Code: [`csrc/`](../csrc) (C++), [`inferlab/kernels/`](../inferlab/kernels) (bindings, NumPy
> references, benchmarks), [`gpu_kernels/`](../gpu_kernels) (CUDA, Triton) · Lab 09 · UI: *Kernels*

A runtime decides *what* to compute each step; kernels decide *how fast* each operation runs. Almost
all LLM time is spent in five kernel families: **GEMM/GEMV**, **attention**, **normalisation and
elementwise ops**, **quantized matmul**, and **sampling**. InferLab implements each on the CPU in
plain, commented C++ with a C ABI. The same algorithms appear on the GPU in `gpu_kernels/`.

```bash
make kernels          # CMake Release build, -O3 -march=native, hardening flags, runs C++ tests
make kernels-test     # the same tests under AddressSanitizer + UndefinedBehaviorSanitizer
./csrc/build/bench_kernels 1024
INFERLAB_KERNELS=cpp inferlab serve    # the live model now uses the C++ kernels
```

## 1. GEMM: the optimisation ladder ([`matmul.cpp`](../csrc/src/matmul.cpp))

| Version | Idea | 1024³ on a 4-core AVX-512 VM |
|---------|------|------------------------------|
| naive i-j-k | textbook triple loop; `B[k][j]` walks a column, so nearly every access misses the cache | ~0.5 GFLOP/s |
| reordered i-k-j | unit-stride inner loop → the compiler emits SIMD | ~13 GFLOP/s |
| tiled + register-blocked | 32×256×64 tiles stay in L1/L2; 4 rows of C per pass reuse each B load 4× | ~17 GFLOP/s |
| tiled + OpenMP | tiles of C are independent → one per thread, no locks | ~85 GFLOP/s |
| OpenBLAS (NumPy) | hand-written assembly micro-kernels, packing, prefetching | ~400 GFLOP/s |

Lesson: data movement, not arithmetic, decides speed. On GPUs the same ladder appears as global
memory → shared-memory tiles → registers → tensor cores (`gpu_kernels/cuda/kernels.cu` `gemm_tiled`,
Triton `tl.dot`). Production engines call cuBLAS/CUTLASS rather than writing their own GEMM.

## 2. GEMV and quantized GEMV ([`matmul.cpp`](../csrc/src/matmul.cpp), [`quant.cpp`](../csrc/src/quant.cpp))

Decode at batch 1 is `y = W·x`: two FLOPs per weight loaded, an arithmetic intensity of 0.5. Time ≈
bytes / bandwidth, so the way to go faster is to read fewer bytes:

| Weights | Bytes / weight | 4096×4096 GEMV (measured) |
|---------|---------------|---------------------------|
| fp32 (AVX2 FMA, 2 accumulators) | 4 | ~0.6–0.7 ms |
| Q8_0 (int8 + scale / 32) | 1.06 | ~0.25 ms |
| Q4_0 (nibbles + scale / 32) | 0.56 | ~0.4 ms (unpacking costs) |

`il_gemv_q8_0` follows ggml. It quantizes the *activation* vector to Q8_0 as well, then multiplies
blocks with **integer** arithmetic using AVX2's `maddubs` trick (`|w| × sign(x, w)`). The two scales
are applied once per block. A scalar fallback keeps it portable (ARM, non-AVX2), and the Docker build
uses that path.

Thread oversubscription is a real production pitfall. NumPy's OpenBLAS threads and our OpenMP
threads both *spin-wait* between calls by default. Interleave them (a NumPy matmul, then a C++
RMSNorm, then a NumPy matmul…) and the two pools fight over the same cores. Measured on a 4-core VM,
decode with `--kernels cpp` went from 7 ms to 34 ms per token. `cpp.py` therefore sets
`OMP_WAIT_POLICY=PASSIVE` before the library loads, and `bench.py` pauses between implementations.
Production servers pin thread counts per process (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`torch.set_num_threads`).

When the C++ backend is active, `Q8_0Linear` / `Q4_0Linear` send decode-sized inputs straight to
these integer GEMVs on the packed weights, instead of dequantizing a full fp32 copy on every call.
That makes quantized decode 2–3× faster. Then try `inferlab generate "Once upon a time" --kernels cpp
--weight-dtype q4_0` on stories15M: it is *still* not faster than fp32. The fp32 LM head
(32000 × 288, tied to the embedding) is the largest matrix read every step, so quantizing only the
transformer layers doesn't reduce the dominant bytes. Real deployments (GGUF Q4_K_M) quantize the
output matrix too. Always profile where the bytes are before optimising.

## 3. Softmax and online softmax ([`elementwise.cpp`](../csrc/src/elementwise.cpp))

Safe softmax needs the max, then the sum of `exp(x − max)`, then a normalisation pass: three passes
over memory. **Online softmax** (Milakov & Gimelshein 2018) keeps a running max `m` and sum `s`, and
rescales `s` whenever the max grows:

```
for x in row:  m' = max(m, x);  s = s·exp(m − m') + exp(x − m');  m = m'
```

That is two passes. On a CPU the difference is small because the row is in cache. On a GPU it saves
a full HBM pass, and it is the recurrence that makes FlashAttention possible.

## 4. FlashAttention ([`attention.cpp`](../csrc/src/attention.cpp))

Naive attention materialises `S = QKᵀ` (`T × T` per head). At 32k tokens that is 4 GiB per head in
fp32. FlashAttention (Dao et al. 2022) streams K/V in tiles and keeps per query row only `(m, l, acc)`:

```
for each K/V tile:
    s     = q·kⱼ / √d                       (for keys in the tile)
    m_new = max(m, max(s));  α = exp(m − m_new)
    l     = l·α + Σ exp(s − m_new)
    acc   = acc·α + Σ exp(s − m_new)·vⱼ
out = acc / l
```

Memory per (head, query tile) becomes O(tile) instead of O(T). On GPUs the tiles live in SRAM, which
cuts HBM traffic from O(T²) to O(T·d). The CPU version tiles queries and keys so each K/V tile is
reused for `block_q` rows, and skips tiles above the causal diagonal. The NumPy version in
[`reference.py`](../inferlab/kernels/reference.py) is the algorithm written out in Python. FA-2
reorders the loops for parallelism; FA-3 (Hopper) adds asynchronous TMA loads, warp specialisation
and FP8.

## 5. PagedAttention decode

`il_paged_attention_decode` computes one query's attention over keys scattered across physical
blocks. It uses the same online softmax, walking the block table. **Every block id is validated
against the pool size before any memory is touched.** A corrupted table must fail with an error, not
read another user's KV cache. vLLM's real kernel adds partitioning across thread blocks for long
contexts (`paged_attention_v2`). FlashInfer and FlashAttention now support paged KV natively.

## 6. Memory-bound elementwise kernels and fusion

RMSNorm, SiLU·mul and RoPE do a handful of FLOPs per byte, so they are memory-bound. The only way to
speed them up is **fusion**: do several operations in one pass over memory. `il_fused_add_rmsnorm`
reads `x` and `residual` once and writes both outputs. vLLM's `fused_add_rms_norm` does the same.
TensorRT-LLM's compiler, `torch.compile` and CUDA Graphs push fusion further, and CUDA Graphs also
remove the per-kernel launch overhead (about 5 µs × hundreds of kernels per decode step).

## 7. FFI safety (Python ↔ C++)

[`cpp.py`](../inferlab/kernels/cpp.py) is a small example of a safe native boundary:

- the library is loaded only from the package directory, never from `LD_LIBRARY_PATH` or the cwd
- every array is converted to C-contiguous memory of the exact dtype, and its shape is checked in
  Python
- the C side re-checks pointers, sizes (with overflow checks), arguments and bounds, and returns a
  status code that Python always checks
- the build uses `-fstack-protector-strong`, `_FORTIFY_SOURCE=2`, full RELRO and a non-executable
  stack, and the C++ tests run under ASan/UBSan in CI
- ctypes releases the GIL, so kernels run in parallel with the server's event loop

## 8. GPU versions

[`gpu_kernels/`](../gpu_kernels/README.md) has a CUDA file (tiled GEMM, warp-shuffle RMSNorm,
FlashAttention forward, PagedAttention decode) and a Triton file (fused softmax, RMSNorm, causal
FlashAttention using tensor cores). Each checks itself against a CPU or PyTorch reference.

---

**Exercises**

1. Add an AVX2 micro-kernel (6×16 accumulators in registers) to `tile_kernel` and climb closer to
   OpenBLAS.
2. Add a Q4_1 format (scale and minimum per block) to `quant.cpp` and the Python quantizer, and
   compare the error with Q4_0.
3. Make `il_flash_attention` process 4 query rows per K load (register blocking for attention).
4. Port `il_paged_attention_decode`'s bounds checks to `gpu_kernels/cuda/kernels.cu`.
