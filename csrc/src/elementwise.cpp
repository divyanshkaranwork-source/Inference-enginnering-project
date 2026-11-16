// Memory-bound kernels: RMSNorm, softmax, SiLU*mul, RoPE.
//
// These do very little arithmetic per byte, so the only way to make them
// faster is to touch memory fewer times. Hence "kernel fusion": e.g.
// fused_add_rmsnorm reads x and residual once and writes both results in the
// same pass, instead of an add kernel + a norm kernel (two extra HBM trips).

#include <cmath>

#include "common.hpp"

namespace {
il_status check_rows(int64_t rows, int64_t cols) {
    int64_t t = 0;
    if (!il::mul_ok(rows, cols, &t) || !il::size_ok(t)) return IL_ERR_SIZE;
    return IL_OK;
}
}  // namespace

extern "C" {

const char* il_version(void) { return "inferlab-kernels 1.0.0"; }

const char* il_simd_name(void) {
#if defined(__AVX512F__)
    return "avx512";
#elif defined(__AVX2__)
    return "avx2";
#elif defined(__ARM_NEON)
    return "neon";
#else
    return "scalar";
#endif
}

int32_t il_has_openmp(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

int32_t il_get_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void il_set_num_threads(int32_t n) {
#ifdef _OPENMP
    if (n > 0 && n <= 1024) omp_set_num_threads(n);
#else
    (void)n;
#endif
}

il_status il_rmsnorm(const float* x, const float* w, float* y, int64_t rows, int64_t dim, float eps) {
    IL_CHECK_PTR(x);
    IL_CHECK_PTR(w);
    IL_CHECK_PTR(y);
    il_status st = check_rows(rows, dim);
    if (st != IL_OK) return st;
    if (!(eps > 0.f)) return IL_ERR_ARG;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * dim;
        float* yr = y + r * dim;
        float ss = 0.f;
        IL_SIMD_REDUCE(ss)
        for (int64_t i = 0; i < dim; ++i) ss += xr[i] * xr[i];
        const float inv = 1.f / std::sqrt(ss / static_cast<float>(dim) + eps);
        IL_SIMD
        for (int64_t i = 0; i < dim; ++i) yr[i] = xr[i] * inv * w[i];
    }
    return IL_OK;
}

// residual += x;  x = rmsnorm(residual) * w     (vLLM: fused_add_rms_norm)
il_status il_fused_add_rmsnorm(float* x, float* residual, const float* w, int64_t rows, int64_t dim, float eps) {
    IL_CHECK_PTR(x);
    IL_CHECK_PTR(residual);
    IL_CHECK_PTR(w);
    il_status st = check_rows(rows, dim);
    if (st != IL_OK) return st;
    if (!(eps > 0.f)) return IL_ERR_ARG;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        float* xr = x + r * dim;
        float* rr = residual + r * dim;
        float ss = 0.f;
        IL_SIMD_REDUCE(ss)
        for (int64_t i = 0; i < dim; ++i) {
            const float v = rr[i] + xr[i];
            rr[i] = v;
            ss += v * v;
        }
        const float inv = 1.f / std::sqrt(ss / static_cast<float>(dim) + eps);
        IL_SIMD
        for (int64_t i = 0; i < dim; ++i) xr[i] = rr[i] * inv * w[i];
    }
    return IL_OK;
}

// Classic safe softmax: 3 passes over the row (max, sum of exp, normalise).
il_status il_softmax_naive(float* x, int64_t rows, int64_t cols) {
    IL_CHECK_PTR(x);
    il_status st = check_rows(rows, cols);
    if (st != IL_OK) return st;
    for (int64_t r = 0; r < rows; ++r) {
        float* v = x + r * cols;
        float m = -INFINITY;
        for (int64_t i = 0; i < cols; ++i) m = std::fmax(m, v[i]);
        if (!std::isfinite(m)) m = 0.f;
        float s = 0.f;
        for (int64_t i = 0; i < cols; ++i) {
            v[i] = std::exp(v[i] - m);
            s += v[i];
        }
        const float inv = s > 0.f ? 1.f / s : 0.f;
        for (int64_t i = 0; i < cols; ++i) v[i] *= inv;
    }
    return IL_OK;
}

// Online softmax (Milakov & Gimelshein 2018): max and normaliser in ONE pass.
// When a new maximum m' appears, rescale the running sum: s = s*exp(m-m') + exp(x-m').
// This recurrence is exactly what lets FlashAttention process keys in tiles.
il_status il_softmax_online(float* x, int64_t rows, int64_t cols) {
    IL_CHECK_PTR(x);
    il_status st = check_rows(rows, cols);
    if (st != IL_OK) return st;
    for (int64_t r = 0; r < rows; ++r) {
        float* v = x + r * cols;
        float m = -INFINITY, s = 0.f;
        for (int64_t i = 0; i < cols; ++i) {
            const float xi = v[i];
            if (xi > m) {
                s = (m == -INFINITY ? 0.f : s * std::exp(m - xi)) + 1.f;
                m = xi;
            } else if (xi != -INFINITY) {
                s += std::exp(xi - m);
            }
        }
        const float inv = s > 0.f ? 1.f / s : 0.f;
        for (int64_t i = 0; i < cols; ++i) v[i] = (m == -INFINITY) ? 0.f : std::exp(v[i] - m) * inv;
    }
    return IL_OK;
}

il_status il_silu_mul(const float* gate, const float* up, float* out, int64_t n) {
    IL_CHECK_PTR(gate);
    IL_CHECK_PTR(up);
    IL_CHECK_PTR(out);
    if (!il::size_ok(n)) return IL_ERR_SIZE;
#pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        const float g = gate[i];
        out[i] = g / (1.f + std::exp(-g)) * up[i];
    }
    return IL_OK;
}

il_status il_rope(float* x, int64_t T, int64_t n_heads, int64_t head_dim, const int64_t* positions, float theta,
                  int32_t interleaved) {
    IL_CHECK_PTR(x);
    IL_CHECK_PTR(positions);
    int64_t total = 0;
    if (!il::mul3_ok(T, n_heads, head_dim, &total) || !il::size_ok(total)) return IL_ERR_SIZE;
    if (head_dim % 2 != 0 || !(theta > 0.f)) return IL_ERR_ARG;
    const int64_t half = head_dim / 2;
#pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; ++t) {
        const double pos = static_cast<double>(positions[t]);
        for (int64_t i = 0; i < half; ++i) {
            const double inv_freq = std::pow(static_cast<double>(theta), -2.0 * double(i) / double(head_dim));
            const float c = static_cast<float>(std::cos(pos * inv_freq));
            const float s = static_cast<float>(std::sin(pos * inv_freq));
            for (int64_t h = 0; h < n_heads; ++h) {
                float* v = x + (t * n_heads + h) * head_dim;
                const int64_t a = interleaved ? 2 * i : i;
                const int64_t b = interleaved ? 2 * i + 1 : i + half;
                const float x0 = v[a], x1 = v[b];
                v[a] = x0 * c - x1 * s;
                v[b] = x0 * s + x1 * c;
            }
        }
    }
    return IL_OK;
}

}  // extern "C"
