// Attention kernels: naive, FlashAttention (tiled, online softmax), PagedAttention.
//
// Naive attention materialises S = Q K^T: O(Tq * Tk) memory per head. At
// 32k context that is 4 GiB per head in fp32 - impossible. FlashAttention
// (Dao et al. 2022) never builds S: it walks over K/V in tiles, keeping per
// query row a running max m, normaliser l and output accumulator acc:
//
//   for each key tile:  s = q.k_j * scale
//                       m' = max(m, max(s));  alpha = exp(m - m')
//                       l  = l*alpha + sum exp(s - m')
//                       acc = acc*alpha + sum exp(s - m') v_j
//   out = acc / l
//
// On a GPU the tiles live in shared memory/registers so HBM traffic drops
// from O(T^2) to O(T * d). On a CPU the same tiling keeps K/V tiles in cache.
//
// PagedAttention (Kwon et al. 2023) is the decode-time variant used by vLLM:
// keys/values live in non-contiguous fixed-size blocks; the kernel follows the
// block table. It bounds-checks every block id - a corrupted table must never
// turn into an out-of-bounds read of another request's KV cache.

#include <algorithm>
#include <cmath>
#include <vector>

#include "common.hpp"

namespace {

il_status check_attn(const float* Q, const float* K, const float* V, const float* O, int64_t Tq, int64_t Tk,
                     int64_t H, int64_t KVH, int64_t hd, int64_t q_offset) {
    IL_CHECK_PTR(Q);
    IL_CHECK_PTR(K);
    IL_CHECK_PTR(V);
    IL_CHECK_PTR(O);
    int64_t t = 0;
    if (!il::mul3_ok(Tq, H, hd, &t) || !il::size_ok(t)) return IL_ERR_SIZE;
    if (!il::mul3_ok(Tk, KVH, hd, &t) || !il::size_ok(t)) return IL_ERR_SIZE;
    if (H % KVH != 0 || q_offset < 0) return IL_ERR_ARG;
    return IL_OK;
}

inline float dot(const float* a, const float* b, int64_t n) {
    float s = 0.f;
    IL_SIMD_REDUCE(s)
    for (int64_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

}  // namespace

extern "C" {

il_status il_attention_naive(const float* Q, const float* K, const float* V, float* O, int64_t Tq, int64_t Tk,
                             int64_t H, int64_t KVH, int64_t hd, int64_t q_offset, int32_t causal) {
    il_status st = check_attn(Q, K, V, O, Tq, Tk, H, KVH, hd, q_offset);
    if (st != IL_OK) return st;
    int64_t n_scores = 0;
    if (!il::mul_ok(Tq, Tk, &n_scores) || !il::size_ok(n_scores)) return IL_ERR_SIZE;
    const int64_t g = H / KVH;
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    std::vector<float> S;  // the full [Tq, Tk] score matrix for one head
    try {
        S.resize(static_cast<size_t>(n_scores));
    } catch (...) {
        return IL_ERR_NOMEM;
    }
    for (int64_t h = 0; h < H; ++h) {
        const int64_t kvh = h / g;
        for (int64_t i = 0; i < Tq; ++i) {
            const float* q = Q + (i * H + h) * hd;
            const int64_t limit = causal ? std::min(Tk, q_offset + i + 1) : Tk;
            float m = -INFINITY;
            for (int64_t j = 0; j < Tk; ++j) {
                float s = -INFINITY;
                if (j < limit) s = dot(q, K + (j * KVH + kvh) * hd, hd) * scale;
                S[i * Tk + j] = s;
                m = std::fmax(m, s);
            }
            float l = 0.f;
            for (int64_t j = 0; j < Tk; ++j) {
                const float p = (j < limit) ? std::exp(S[i * Tk + j] - m) : 0.f;
                S[i * Tk + j] = p;
                l += p;
            }
            float* o = O + (i * H + h) * hd;
            std::fill(o, o + hd, 0.f);
            const float inv = l > 0.f ? 1.f / l : 0.f;
            for (int64_t j = 0; j < limit; ++j) {
                const float p = S[i * Tk + j] * inv;
                const float* v = V + (j * KVH + kvh) * hd;
                IL_SIMD
                for (int64_t d = 0; d < hd; ++d) o[d] += p * v[d];
            }
        }
    }
    return IL_OK;
}

il_status il_flash_attention(const float* Q, const float* K, const float* V, float* O, int64_t Tq, int64_t Tk,
                             int64_t H, int64_t KVH, int64_t hd, int64_t q_offset, int32_t causal, int64_t block_q,
                             int64_t block_k) {
    il_status st = check_attn(Q, K, V, O, Tq, Tk, H, KVH, hd, q_offset);
    if (st != IL_OK) return st;
    if (block_q <= 0 || block_k <= 0 || block_q > 4096 || block_k > 4096 || hd > 1024) return IL_ERR_ARG;
    const int64_t g = H / KVH;
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    const int64_t n_qt = (Tq + block_q - 1) / block_q;

#pragma omp parallel
    {
        // per-thread scratch: O(block_q * (hd + block_k)) - independent of sequence length
        std::vector<float> acc(static_cast<size_t>(block_q * hd));
        std::vector<float> m(static_cast<size_t>(block_q)), l(static_cast<size_t>(block_q));
        std::vector<float> s(static_cast<size_t>(block_k));
#pragma omp for collapse(2) schedule(dynamic)
        for (int64_t h = 0; h < H; ++h)
            for (int64_t qt = 0; qt < n_qt; ++qt) {
                const int64_t kvh = h / g;
                const int64_t i0 = qt * block_q, i1 = std::min(Tq, i0 + block_q);
                std::fill(acc.begin(), acc.end(), 0.f);
                std::fill(m.begin(), m.end(), -INFINITY);
                std::fill(l.begin(), l.end(), 0.f);
                // keys beyond the last query's causal limit are never needed
                const int64_t k_end = causal ? std::min(Tk, q_offset + i1) : Tk;
                for (int64_t j0 = 0; j0 < k_end; j0 += block_k) {
                    const int64_t j1 = std::min(k_end, j0 + block_k);
                    for (int64_t i = i0; i < i1; ++i) {
                        const int64_t r = i - i0;
                        const int64_t limit = causal ? std::min(j1, q_offset + i + 1) : j1;
                        if (limit <= j0) continue;  // whole tile masked for this row
                        const float* q = Q + (i * H + h) * hd;
                        float tile_max = -INFINITY;
                        for (int64_t j = j0; j < limit; ++j) {
                            const float sc = dot(q, K + (j * KVH + kvh) * hd, hd) * scale;
                            s[j - j0] = sc;
                            tile_max = std::fmax(tile_max, sc);
                        }
                        const float m_new = std::fmax(m[r], tile_max);
                        const float alpha = (m[r] == -INFINITY) ? 0.f : std::exp(m[r] - m_new);
                        float* a = acc.data() + r * hd;
                        IL_SIMD
                        for (int64_t d = 0; d < hd; ++d) a[d] *= alpha;
                        float lsum = 0.f;
                        for (int64_t j = j0; j < limit; ++j) {
                            const float p = std::exp(s[j - j0] - m_new);
                            lsum += p;
                            const float* v = V + (j * KVH + kvh) * hd;
                            IL_SIMD
                            for (int64_t d = 0; d < hd; ++d) a[d] += p * v[d];
                        }
                        l[r] = l[r] * alpha + lsum;
                        m[r] = m_new;
                    }
                }
                for (int64_t i = i0; i < i1; ++i) {
                    const int64_t r = i - i0;
                    float* o = O + (i * H + h) * hd;
                    const float inv = l[r] > 0.f ? 1.f / l[r] : 0.f;
                    for (int64_t d = 0; d < hd; ++d) o[d] = acc[r * hd + d] * inv;
                }
            }
    }
    return IL_OK;
}

il_status il_paged_attention_decode(const float* q, const float* k_cache, const float* v_cache,
                                    const int32_t* block_table, int64_t n_table, int64_t context_len,
                                    int64_t num_blocks, int64_t block_size, int64_t H, int64_t KVH, int64_t hd,
                                    float* out) {
    IL_CHECK_PTR(q);
    IL_CHECK_PTR(k_cache);
    IL_CHECK_PTR(v_cache);
    IL_CHECK_PTR(block_table);
    IL_CHECK_PTR(out);
    IL_CHECK_POS(n_table);
    IL_CHECK_POS(context_len);
    IL_CHECK_POS(block_size);
    int64_t t = 0;
    if (!il::mul3_ok(num_blocks, block_size, KVH, &t) || !il::mul_ok(t, hd, &t) || !il::size_ok(t))
        return IL_ERR_SIZE;
    if (!il::mul_ok(H, hd, &t) || H % KVH != 0 || hd > 1024) return IL_ERR_ARG;
    const int64_t needed_blocks = (context_len + block_size - 1) / block_size;
    if (needed_blocks > n_table) return IL_ERR_ARG;
    for (int64_t b = 0; b < needed_blocks; ++b)  // validate before touching memory
        if (block_table[b] < 0 || block_table[b] >= num_blocks) return IL_ERR_BOUNDS;

    const int64_t g = H / KVH;
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    const int64_t block_stride = block_size * KVH * hd;
#pragma omp parallel for schedule(static)
    for (int64_t h = 0; h < H; ++h) {
        const int64_t kvh = h / g;
        const float* qh = q + h * hd;
        std::vector<float> acc(static_cast<size_t>(hd), 0.f);
        float m = -INFINITY, l = 0.f;
        for (int64_t b = 0; b < needed_blocks; ++b) {
            const float* kb = k_cache + int64_t(block_table[b]) * block_stride;
            const float* vb = v_cache + int64_t(block_table[b]) * block_stride;
            const int64_t n_in_block = std::min(block_size, context_len - b * block_size);
            for (int64_t j = 0; j < n_in_block; ++j) {
                const float s = dot(qh, kb + (j * KVH + kvh) * hd, hd) * scale;
                const float m_new = std::fmax(m, s);
                const float alpha = (m == -INFINITY) ? 0.f : std::exp(m - m_new);
                const float p = std::exp(s - m_new);
                const float* v = vb + (j * KVH + kvh) * hd;
                IL_SIMD
                for (int64_t d = 0; d < hd; ++d) acc[d] = acc[d] * alpha + p * v[d];
                l = l * alpha + p;
                m = m_new;
            }
        }
        const float inv = l > 0.f ? 1.f / l : 0.f;
        for (int64_t d = 0; d < hd; ++d) out[h * hd + d] = acc[d] * inv;
    }
    return IL_OK;
}

}  // extern "C"
