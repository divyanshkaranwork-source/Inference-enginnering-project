/*
 * InferLab CPU kernels - a C ABI so Python (ctypes), C++ tests and benchmarks
 * can all call the same code.
 *
 * Conventions
 *   - all matrices are row-major, float32 unless stated otherwise
 *   - every function validates its arguments and returns an il_status
 *     (0 = OK, negative = error). Callers MUST check it.
 *   - sizes are int64_t; products are overflow-checked before allocating
 *
 * Kernel map (what each one teaches)
 *   matmul_naive / reordered / tiled / parallel   cache locality, tiling, threads
 *   gemv / gemv_q8_0 / gemv_q4_0                  decode is memory-bound; quantization
 *   softmax_naive / softmax_online                3-pass vs 2-pass (online) softmax
 *   rmsnorm / fused_add_rmsnorm / silu_mul / rope  memory-bound elementwise fusion
 *   attention_naive / flash_attention             O(N^2) memory vs tiled online softmax
 *   paged_attention_decode                        vLLM's block-table attention
 */
#ifndef INFERLAB_KERNELS_H
#define INFERLAB_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define IL_API __declspec(dllexport)
#else
#define IL_API __attribute__((visibility("default")))
#endif

typedef int32_t il_status;
#define IL_OK 0
#define IL_ERR_NULL (-1)     /* a required pointer was NULL            */
#define IL_ERR_SIZE (-2)     /* a dimension was <= 0 or too large      */
#define IL_ERR_ARG (-3)      /* inconsistent arguments                 */
#define IL_ERR_BOUNDS (-4)   /* an index (e.g. block id) out of range  */
#define IL_ERR_NOMEM (-5)    /* allocation failed                      */

/* ---- info ---------------------------------------------------------- */
IL_API const char* il_version(void);
IL_API const char* il_simd_name(void);
IL_API int32_t il_has_openmp(void);
IL_API int32_t il_get_max_threads(void);
IL_API void il_set_num_threads(int32_t n);

/* ---- GEMM: C[M,N] = A[M,K] @ B[K,N] -------------------------------- */
IL_API il_status il_matmul_naive(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K);
IL_API il_status il_matmul_reordered(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K);
IL_API il_status il_matmul_tiled(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K);
IL_API il_status il_matmul_parallel(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K);

/* ---- GEMV: y[N] = W[N,K] @ x[K]  (the decode-time linear layer) ----- */
IL_API il_status il_gemv(const float* W, const float* x, float* y, int64_t N, int64_t K);

/* ---- quantization (llama.cpp-style blocks of 32) ------------------- */
#define IL_QK 32
IL_API il_status il_quantize_q8_0(const float* x, int64_t n, int8_t* qs, float* d);
IL_API il_status il_quantize_q4_0(const float* x, int64_t n, uint8_t* qs, float* d);
IL_API il_status il_gemv_q8_0(const int8_t* Wq, const float* Wd, const float* x, float* y, int64_t N, int64_t K);
IL_API il_status il_gemv_q4_0(const uint8_t* Wq, const float* Wd, const float* x, float* y, int64_t N, int64_t K);

/* ---- elementwise / normalisation ------------------------------------ */
IL_API il_status il_rmsnorm(const float* x, const float* w, float* y, int64_t rows, int64_t dim, float eps);
IL_API il_status il_fused_add_rmsnorm(float* x, float* residual, const float* w, int64_t rows, int64_t dim, float eps);
IL_API il_status il_softmax_naive(float* x, int64_t rows, int64_t cols);
IL_API il_status il_softmax_online(float* x, int64_t rows, int64_t cols);
IL_API il_status il_silu_mul(const float* gate, const float* up, float* out, int64_t n);
IL_API il_status il_rope(float* x, int64_t T, int64_t n_heads, int64_t head_dim, const int64_t* positions,
                         float theta, int32_t interleaved);

/* ---- attention ------------------------------------------------------
 * Q[Tq, H, hd], K/V[Tk, KVH, hd], O[Tq, H, hd]. Query i is at absolute
 * position q_offset + i and (if causal) sees keys 0..q_offset+i.          */
IL_API il_status il_attention_naive(const float* Q, const float* K, const float* V, float* O, int64_t Tq,
                                    int64_t Tk, int64_t H, int64_t KVH, int64_t hd, int64_t q_offset,
                                    int32_t causal);
IL_API il_status il_flash_attention(const float* Q, const float* K, const float* V, float* O, int64_t Tq,
                                    int64_t Tk, int64_t H, int64_t KVH, int64_t hd, int64_t q_offset,
                                    int32_t causal, int64_t block_q, int64_t block_k);

/* q[H, hd]; k_cache/v_cache[num_blocks, block_size, KVH, hd];
 * block_table[n_table] maps logical -> physical blocks; out[H, hd]        */
IL_API il_status il_paged_attention_decode(const float* q, const float* k_cache, const float* v_cache,
                                           const int32_t* block_table, int64_t n_table, int64_t context_len,
                                           int64_t num_blocks, int64_t block_size, int64_t H, int64_t KVH,
                                           int64_t hd, float* out);

#ifdef __cplusplus
}
#endif
#endif /* INFERLAB_KERNELS_H */
