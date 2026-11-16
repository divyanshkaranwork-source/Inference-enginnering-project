"""Lab 09 - Kernels: tiling, SIMD, threads, FlashAttention, quantized GEMV.

Run:  make kernels && python labs/09_kernels.py
Read: docs/03-kernels.md
"""

from _common import explain, header, table

from inferlab.kernels import bench, cpp

print(f"[kernels] {cpp.info()}")
if not cpp.available():
    print("  C++ library not built - run `make kernels` first (NumPy rows still run).")

header("1. The GEMM optimisation ladder (512 x 512 x 512)")
r = bench.bench_matmul(512)
table([[x["name"], f"{x['ms']:.2f}", f"{x['gflops']:.1f}"] for x in r["results"]], ["implementation", "ms", "GFLOP/s"])
explain("Loop order alone (i-k-j: unit-stride inner loop) unlocks SIMD; tiling keeps blocks in cache; register "
        "blocking reuses each load 4x; OpenMP uses every core. BLAS libraries go further with hand-written micro-"
        "kernels - on GPUs that is cuBLAS/CUTLASS with tensor cores.")

header("2. Decode GEMV: bytes are everything (4096 x 4096 = one Llama-2-7B projection)")
r = bench.bench_gemv(4096, 4096)
table([[x["name"], f"{x['ms']:.3f}", f"{x['weight_mb']:.1f}", f"{x['gbps']:.1f}", f"{x['rel_err']:.1e}"]
       for x in r["results"]], ["implementation", "ms", "MiB read", "GB/s", "rel err"])
explain("Each weight is used once, so time ~ bytes / bandwidth. Q8_0 and Q4_0 read 3.8x / 7x fewer bytes than fp32 "
        "and win despite unpacking - IF the unpacking is vectorised (see quant.cpp's AVX2 maddubs trick).")

header("3. Attention: naive vs FlashAttention (seq 1024, 8 heads, d=64)")
r = bench.bench_attention(1024, 8, 64)
table([[x["name"], f"{x['ms']:.1f}", f"{x['scratch_mb']:.2f}", f"{x['max_err']:.1e}"] for x in r["results"]],
      ["implementation", "ms", "scratch MiB", "max err"])
explain("FlashAttention never materialises the T x T score matrix: it streams K/V tiles and keeps a running max "
        "and sum (online softmax). Scratch memory is O(tile), not O(T^2) - the only way 128k contexts fit.")
