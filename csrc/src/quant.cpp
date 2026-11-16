// llama.cpp-style block quantization: Q8_0 and Q4_0 with integer dot products.
//
// Q8_0 block (32 weights): int8 q[32] + scale d       -> 34 bytes (ggml: fp16 d)
// Q4_0 block (32 weights): 16 bytes of nibbles + d     -> 18 bytes (4.5 bits/weight)
//
// GEMV with quantized weights follows ggml: the *activation* vector is
// quantized to Q8_0 on the fly, then each block pair is multiplied with
// integer arithmetic and scaled once:   y += d_w * d_x * sum_i(q_w[i] * q_x[i])
// Reading 4.5 bits instead of 32 bits per weight is a ~7x reduction in memory
// traffic - which, for memory-bound decode, is almost a 7x speedup.
// (Scales are float32 here for simplicity; ggml stores them as fp16.)

#include <algorithm>
#include <cmath>
#include <vector>

#include "common.hpp"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define IL_HAVE_AVX2 1
#endif

namespace {

#ifdef IL_HAVE_AVX2
// llama.cpp's trick for a signed int8 dot product with AVX2:
// maddubs needs (unsigned, signed) operands, so use |w| and x*sign(w).
inline __m256 dot_i8_to_float(__m256i w, __m256i x) {
    const __m256i aw = _mm256_sign_epi8(w, w);
    const __m256i sx = _mm256_sign_epi8(x, w);
    const __m256i p16 = _mm256_maddubs_epi16(aw, sx);               // 16 x int16
    const __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));  // 8 x int32
    return _mm256_cvtepi32_ps(p32);
}

inline float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

// 16 packed bytes -> 32 signed int8 in [-8, 7]: low nibbles = elements 0..15
inline __m256i unpack_q4(const uint8_t* p) {
    const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(raw, mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
    const __m256i both = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    return _mm256_sub_epi8(both, _mm256_set1_epi8(8));
}
#endif

inline il_status check_q(int64_t n) {
    if (n <= 0 || n % IL_QK != 0 || !il::size_ok(n)) return IL_ERR_SIZE;
    return IL_OK;
}

void quantize_block_q8(const float* x, int8_t* q, float* d) {
    float amax = 0.f;
    for (int i = 0; i < IL_QK; ++i) amax = std::fmax(amax, std::fabs(x[i]));
    const float s = amax / 127.f;
    const float inv = s != 0.f ? 1.f / s : 0.f;
    for (int i = 0; i < IL_QK; ++i) {
        float v = std::nearbyint(x[i] * inv);
        v = std::fmin(127.f, std::fmax(-127.f, v));
        q[i] = static_cast<int8_t>(v);
    }
    *d = s;
}

}  // namespace

extern "C" {

il_status il_quantize_q8_0(const float* x, int64_t n, int8_t* qs, float* d) {
    IL_CHECK_PTR(x);
    IL_CHECK_PTR(qs);
    IL_CHECK_PTR(d);
    il_status st = check_q(n);
    if (st != IL_OK) return st;
    for (int64_t b = 0; b < n / IL_QK; ++b) quantize_block_q8(x + b * IL_QK, qs + b * IL_QK, d + b);
    return IL_OK;
}

il_status il_quantize_q4_0(const float* x, int64_t n, uint8_t* qs, float* d) {
    IL_CHECK_PTR(x);
    IL_CHECK_PTR(qs);
    IL_CHECK_PTR(d);
    il_status st = check_q(n);
    if (st != IL_OK) return st;
    for (int64_t b = 0; b < n / IL_QK; ++b) {
        const float* xb = x + b * IL_QK;
        float amax = 0.f, mx = 0.f;  // value with the largest magnitude (keeps its sign)
        for (int i = 0; i < IL_QK; ++i)
            if (std::fabs(xb[i]) > amax) {
                amax = std::fabs(xb[i]);
                mx = xb[i];
            }
        const float s = mx / -8.f;
        const float inv = s != 0.f ? 1.f / s : 0.f;
        d[b] = s;
        uint8_t* qb = qs + b * (IL_QK / 2);
        for (int i = 0; i < IL_QK / 2; ++i) {
            const int lo = std::min(15, std::max(0, static_cast<int>(std::floor(xb[i] * inv + 8.5f))));
            const int hi =
                std::min(15, std::max(0, static_cast<int>(std::floor(xb[i + IL_QK / 2] * inv + 8.5f))));
            qb[i] = static_cast<uint8_t>(lo | (hi << 4));
        }
    }
    return IL_OK;
}

il_status il_gemv_q8_0(const int8_t* Wq, const float* Wd, const float* x, float* y, int64_t N, int64_t K) {
    IL_CHECK_PTR(Wq);
    IL_CHECK_PTR(Wd);
    IL_CHECK_PTR(x);
    IL_CHECK_PTR(y);
    int64_t t = 0;
    if (!il::mul_ok(N, K, &t) || !il::size_ok(t) || check_q(K) != IL_OK) return IL_ERR_SIZE;
    const int64_t nb = K / IL_QK;
    std::vector<int8_t> xq(static_cast<size_t>(K));
    std::vector<float> xd(static_cast<size_t>(nb));
    for (int64_t b = 0; b < nb; ++b) quantize_block_q8(x + b * IL_QK, xq.data() + b * IL_QK, xd.data() + b);
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        const int8_t* wq = Wq + n * K;
        const float* wd = Wd + n * nb;
#ifdef IL_HAVE_AVX2
        __m256 vacc = _mm256_setzero_ps();
        for (int64_t b = 0; b < nb; ++b) {
            const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wq + b * IL_QK));
            const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq.data() + b * IL_QK));
            vacc = _mm256_fmadd_ps(_mm256_set1_ps(wd[b] * xd[b]), dot_i8_to_float(w, xv), vacc);
        }
        y[n] = hsum(vacc);
#else
        float acc = 0.f;
        for (int64_t b = 0; b < nb; ++b) {
            int32_t isum = 0;
            const int8_t* wb = wq + b * IL_QK;
            const int8_t* xb = xq.data() + b * IL_QK;
            IL_SIMD_REDUCE(isum)
            for (int i = 0; i < IL_QK; ++i) isum += int32_t(wb[i]) * int32_t(xb[i]);
            acc += wd[b] * xd[b] * static_cast<float>(isum);
        }
        y[n] = acc;
#endif
    }
    return IL_OK;
}

il_status il_gemv_q4_0(const uint8_t* Wq, const float* Wd, const float* x, float* y, int64_t N, int64_t K) {
    IL_CHECK_PTR(Wq);
    IL_CHECK_PTR(Wd);
    IL_CHECK_PTR(x);
    IL_CHECK_PTR(y);
    int64_t t = 0;
    if (!il::mul_ok(N, K, &t) || !il::size_ok(t) || check_q(K) != IL_OK) return IL_ERR_SIZE;
    const int64_t nb = K / IL_QK;
    std::vector<int8_t> xq(static_cast<size_t>(K));
    std::vector<float> xd(static_cast<size_t>(nb));
    for (int64_t b = 0; b < nb; ++b) quantize_block_q8(x + b * IL_QK, xq.data() + b * IL_QK, xd.data() + b);
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        const uint8_t* wq = Wq + n * (K / 2);
        const float* wd = Wd + n * nb;
#ifdef IL_HAVE_AVX2
        __m256 vacc = _mm256_setzero_ps();
        for (int64_t b = 0; b < nb; ++b) {
            const __m256i w = unpack_q4(wq + b * (IL_QK / 2));
            const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq.data() + b * IL_QK));
            vacc = _mm256_fmadd_ps(_mm256_set1_ps(wd[b] * xd[b]), dot_i8_to_float(w, xv), vacc);
        }
        y[n] = hsum(vacc);
#else
        float acc = 0.f;
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t* wb = wq + b * (IL_QK / 2);
            const int8_t* xb = xq.data() + b * IL_QK;
            int32_t isum = 0;
            for (int i = 0; i < IL_QK / 2; ++i) {
                const int lo = int(wb[i] & 0x0F) - 8;  // element i
                const int hi = int(wb[i] >> 4) - 8;    // element i + 16
                isum += lo * int32_t(xb[i]) + hi * int32_t(xb[i + IL_QK / 2]);
            }
            acc += wd[b] * xd[b] * static_cast<float>(isum);
        }
        y[n] = acc;
#endif
    }
    return IL_OK;
}

}  // extern "C"
