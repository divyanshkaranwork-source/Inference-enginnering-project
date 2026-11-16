// GEMM and GEMV - the kernels that dominate LLM inference time.
//
// Four GEMM versions show the classic optimisation ladder. On one CPU core
// (M=N=K=512) you will typically see something like:
//   naive (i-j-k)          ~0.3-1 GFLOP/s   strided B access, cache misses
//   reordered (i-k-j)      ~3-10 GFLOP/s    unit-stride inner loop -> SIMD
//   tiled                  ~10-30 GFLOP/s   blocks stay in L1/L2 cache
//   parallel (tiled + OMP) x cores
// and NumPy/OpenBLAS (hand-written assembly micro-kernels) beats them all.
// GPUs apply the same ideas: tiles in shared memory, registers per thread,
// tensor cores for the inner product (see gpu_kernels/).

#include <algorithm>
#include <cstring>

#include "common.hpp"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define IL_HAVE_AVX2 1
#endif

namespace {

il_status check_gemm(const float* A, const float* B, const float* C, int64_t M, int64_t N, int64_t K) {
    IL_CHECK_PTR(A);
    IL_CHECK_PTR(B);
    IL_CHECK_PTR(C);
    int64_t t = 0;
    if (!il::mul_ok(M, K, &t) || !il::size_ok(t)) return IL_ERR_SIZE;
    if (!il::mul_ok(K, N, &t) || !il::size_ok(t)) return IL_ERR_SIZE;
    if (!il::mul_ok(M, N, &t) || !il::size_ok(t)) return IL_ERR_SIZE;
    return IL_OK;
}

// Tile sizes: a TK x TN panel of B (~64 KiB) is reused across TM rows of A.
constexpr int64_t TM = 32;
constexpr int64_t TN = 256;
constexpr int64_t TK = 64;

// C[i0:i1, j0:j1] += A[i0:i1, k0:k1] @ B[k0:k1, j0:j1]
// Register blocking: 4 rows of C are updated per pass, so every element of
// B loaded from cache is used 4 times (4 FMAs per load instead of 1).
inline void tile_kernel(const float* A, const float* B, float* C, int64_t N, int64_t K, int64_t i0,
                        int64_t i1, int64_t j0, int64_t j1, int64_t k0, int64_t k1) {
    int64_t i = i0;
    for (; i + 4 <= i1; i += 4) {
        float* c0 = C + i * N;
        float* c1 = c0 + N;
        float* c2 = c1 + N;
        float* c3 = c2 + N;
        const float* a = A + i * K;
        for (int64_t k = k0; k < k1; ++k) {
            const float a0 = a[k], a1 = a[K + k], a2 = a[2 * K + k], a3 = a[3 * K + k];
            const float* b = B + k * N;
            IL_SIMD
            for (int64_t j = j0; j < j1; ++j) {
                const float bj = b[j];
                c0[j] += a0 * bj;
                c1[j] += a1 * bj;
                c2[j] += a2 * bj;
                c3[j] += a3 * bj;
            }
        }
    }
    for (; i < i1; ++i) {  // leftover rows
        float* c = C + i * N;
        const float* a = A + i * K;
        for (int64_t k = k0; k < k1; ++k) {
            const float aik = a[k];
            const float* b = B + k * N;
            IL_SIMD
            for (int64_t j = j0; j < j1; ++j) c[j] += aik * b[j];
        }
    }
}

// Unblocked i-k-j loop (only the loop order changes vs. naive).
inline void ikj(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K) {
    for (int64_t i = 0; i < M; ++i) {
        float* c = C + i * N;
        for (int64_t k = 0; k < K; ++k) {
            const float aik = A[i * K + k];
            const float* b = B + k * N;
            IL_SIMD
            for (int64_t j = 0; j < N; ++j) c[j] += aik * b[j];
        }
    }
}

}  // namespace

extern "C" {

il_status il_matmul_naive(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K) {
    il_status st = check_gemm(A, B, C, M, N, K);
    if (st != IL_OK) return st;
    for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int64_t k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];  // B walks a column: stride N
            C[i * N + j] = acc;
        }
    return IL_OK;
}

il_status il_matmul_reordered(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K) {
    il_status st = check_gemm(A, B, C, M, N, K);
    if (st != IL_OK) return st;
    std::memset(C, 0, sizeof(float) * static_cast<size_t>(M * N));
    ikj(A, B, C, M, N, K);  // i-k-j order: unit-stride inner loop the compiler can vectorise
    return IL_OK;
}

il_status il_matmul_tiled(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K) {
    il_status st = check_gemm(A, B, C, M, N, K);
    if (st != IL_OK) return st;
    std::memset(C, 0, sizeof(float) * static_cast<size_t>(M * N));
    for (int64_t j0 = 0; j0 < N; j0 += TN)
        for (int64_t k0 = 0; k0 < K; k0 += TK)
            for (int64_t i0 = 0; i0 < M; i0 += TM)
                tile_kernel(A, B, C, N, K, i0, std::min(i0 + TM, M), j0, std::min(j0 + TN, N), k0,
                            std::min(k0 + TK, K));
    return IL_OK;
}

il_status il_matmul_parallel(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K) {
    il_status st = check_gemm(A, B, C, M, N, K);
    if (st != IL_OK) return st;
    std::memset(C, 0, sizeof(float) * static_cast<size_t>(M * N));
    const int64_t mt = (M + TM - 1) / TM, nt = (N + TN - 1) / TN;
    // each thread owns whole C tiles -> no write conflicts, no atomics
#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t ti = 0; ti < mt; ++ti)
        for (int64_t tj = 0; tj < nt; ++tj) {
            const int64_t i0 = ti * TM, j0 = tj * TN;
            for (int64_t k0 = 0; k0 < K; k0 += TK)
                tile_kernel(A, B, C, N, K, i0, std::min(i0 + TM, M), j0, std::min(j0 + TN, N), k0,
                            std::min(k0 + TK, K));
        }
    return IL_OK;
}

// y = W x. Every weight is used exactly once -> 2 FLOPs per 4 bytes loaded:
// arithmetic intensity 0.5 FLOP/byte. Speed is set by memory bandwidth, not
// by how clever the arithmetic is. This is the decode bottleneck.
il_status il_gemv(const float* W, const float* x, float* y, int64_t N, int64_t K) {
    IL_CHECK_PTR(W);
    IL_CHECK_PTR(x);
    IL_CHECK_PTR(y);
    int64_t t = 0;
    if (!il::mul_ok(N, K, &t) || !il::size_ok(t)) return IL_ERR_SIZE;
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        const float* w = W + n * K;
        float acc = 0.f;
#ifdef IL_HAVE_AVX2
        __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
        int64_t k = 0;
        for (; k + 16 <= K; k += 16) {  // two independent accumulators hide FMA latency
            v0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + k), _mm256_loadu_ps(x + k), v0);
            v1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + k + 8), _mm256_loadu_ps(x + k + 8), v1);
        }
        v0 = _mm256_add_ps(v0, v1);
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, v0);
        for (int l = 0; l < 8; ++l) acc += lanes[l];
        for (; k < K; ++k) acc += w[k] * x[k];
#else
        IL_SIMD_REDUCE(acc)
        for (int64_t k = 0; k < K; ++k) acc += w[k] * x[k];
#endif
        y[n] = acc;
    }
    return IL_OK;
}

}  // extern "C"
