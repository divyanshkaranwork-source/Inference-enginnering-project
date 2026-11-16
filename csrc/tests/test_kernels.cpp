// Self-contained C++ tests for the kernels (no framework needed).
// Run under AddressSanitizer/UBSan in CI:  make kernels-test-asan

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "inferlab_kernels.h"

static int g_failures = 0;

#define EXPECT(cond, msg)                                              \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ++g_failures;                                              \
        }                                                              \
    } while (0)

static std::vector<float> randv(size_t n, unsigned seed, float scale = 1.f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
    return m;
}

static void test_matmul() {
    const int64_t M = 37, N = 300, K = 129;  // deliberately not multiples of tile sizes
    auto A = randv(M * K, 1), B = randv(K * N, 2);
    std::vector<float> ref(M * N), c(M * N);
    EXPECT(il_matmul_naive(A.data(), B.data(), ref.data(), M, N, K) == IL_OK, "naive status");
    for (auto fn : {il_matmul_reordered, il_matmul_tiled, il_matmul_parallel}) {
        EXPECT(fn(A.data(), B.data(), c.data(), M, N, K) == IL_OK, "gemm status");
        EXPECT(max_abs_diff(ref, c) < 1e-3f, "gemm mismatch");
    }
    // gemv == matmul with N=1 on the transposed problem
    std::vector<float> W = randv(N * K, 3), x = randv(K, 4), y(N), yref(N, 0.f);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) yref[n] += W[n * K + k] * x[k];
    EXPECT(il_gemv(W.data(), x.data(), y.data(), N, K) == IL_OK, "gemv status");
    EXPECT(max_abs_diff(yref, y) < 1e-3f, "gemv mismatch");
}

static void test_errors() {
    float a = 0.f;
    EXPECT(il_matmul_naive(nullptr, &a, &a, 1, 1, 1) == IL_ERR_NULL, "null check");
    EXPECT(il_matmul_naive(&a, &a, &a, 0, 1, 1) == IL_ERR_SIZE, "size check");
    EXPECT(il_matmul_naive(&a, &a, &a, int64_t(1) << 40, int64_t(1) << 40, 1) == IL_ERR_SIZE, "overflow check");
    EXPECT(il_rmsnorm(&a, &a, &a, 1, 1, 0.f) == IL_ERR_ARG, "eps check");
    std::vector<float> q(8), kc(4 * 2 * 1 * 8), out(8);
    int32_t bad_table[2] = {0, 99};  // 99 >= num_blocks -> must be rejected
    EXPECT(il_paged_attention_decode(q.data(), kc.data(), kc.data(), bad_table, 2, 3, 4, 2, 1, 1, 8, out.data()) ==
               IL_ERR_BOUNDS,
           "paged bounds check");
    int32_t neg_table[1] = {-1};
    EXPECT(il_paged_attention_decode(q.data(), kc.data(), kc.data(), neg_table, 1, 1, 4, 2, 1, 1, 8, out.data()) ==
               IL_ERR_BOUNDS,
           "paged negative id");
}

static void test_softmax_and_norm() {
    const int64_t R = 5, C = 77;
    auto x = randv(R * C, 5, 4.f);
    auto a = x, b = x;
    EXPECT(il_softmax_naive(a.data(), R, C) == IL_OK, "softmax naive");
    EXPECT(il_softmax_online(b.data(), R, C) == IL_OK, "softmax online");
    EXPECT(max_abs_diff(a, b) < 1e-6f, "online softmax mismatch");
    float row_sum = 0.f;
    for (int64_t i = 0; i < C; ++i) row_sum += a[i];
    EXPECT(std::fabs(row_sum - 1.f) < 1e-5f, "softmax sums to 1");

    auto w = randv(C, 6), y = std::vector<float>(R * C);
    EXPECT(il_rmsnorm(x.data(), w.data(), y.data(), R, C, 1e-5f) == IL_OK, "rmsnorm");
    auto xr = x;
    auto res = randv(R * C, 7);
    auto res2 = res;
    EXPECT(il_fused_add_rmsnorm(xr.data(), res.data(), w.data(), R, C, 1e-5f) == IL_OK, "fused");
    for (int64_t i = 0; i < R * C; ++i) res2[i] += x[i];
    std::vector<float> y2(R * C);
    il_rmsnorm(res2.data(), w.data(), y2.data(), R, C, 1e-5f);
    EXPECT(max_abs_diff(xr, y2) < 1e-5f && max_abs_diff(res, res2) < 1e-6f, "fused add rmsnorm");
}

static void test_attention() {
    const int64_t Tq = 9, Tk = 40, H = 4, KVH = 2, hd = 16, off = Tk - Tq;
    auto Q = randv(Tq * H * hd, 8), K = randv(Tk * KVH * hd, 9), V = randv(Tk * KVH * hd, 10);
    std::vector<float> o1(Tq * H * hd), o2(Tq * H * hd);
    for (int causal = 0; causal <= 1; ++causal) {
        EXPECT(il_attention_naive(Q.data(), K.data(), V.data(), o1.data(), Tq, Tk, H, KVH, hd, off, causal) == IL_OK,
               "naive attn");
        for (int64_t bk : {1, 7, 16, 64}) {
            EXPECT(il_flash_attention(Q.data(), K.data(), V.data(), o2.data(), Tq, Tk, H, KVH, hd, off, causal, 4,
                                      bk) == IL_OK,
                   "flash attn");
            EXPECT(max_abs_diff(o1, o2) < 1e-4f, "flash != naive");
        }
    }
    // paged decode == naive attention over the last query
    const int64_t bs = 8, nblocks = 10, ctx = 29;
    std::vector<float> kc(nblocks * bs * KVH * hd, 0.f), vc(nblocks * bs * KVH * hd, 0.f);
    int32_t table[4] = {7, 2, 9, 0};
    for (int64_t p = 0; p < ctx; ++p)
        for (int64_t e = 0; e < KVH * hd; ++e) {
            const int64_t slot = (int64_t(table[p / bs]) * bs + p % bs) * KVH * hd + e;
            kc[slot] = K[p * KVH * hd + e];
            vc[slot] = V[p * KVH * hd + e];
        }
    std::vector<float> q1(Q.begin(), Q.begin() + H * hd), op(H * hd), on(H * hd);
    EXPECT(il_paged_attention_decode(q1.data(), kc.data(), vc.data(), table, 4, ctx, nblocks, bs, H, KVH, hd,
                                     op.data()) == IL_OK,
           "paged status");
    il_attention_naive(q1.data(), K.data(), V.data(), on.data(), 1, ctx, H, KVH, hd, ctx - 1, 1);
    EXPECT(max_abs_diff(op, on) < 1e-4f, "paged != naive");
}

static void test_quant() {
    const int64_t N = 33, K = 256;
    auto W = randv(N * K, 11), x = randv(K, 12);
    std::vector<float> yref(N, 0.f), y(N);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) yref[n] += W[n * K + k] * x[k];
    std::vector<int8_t> q8(N * K);
    std::vector<float> d(N * K / IL_QK);
    EXPECT(il_quantize_q8_0(W.data(), N * K, q8.data(), d.data()) == IL_OK, "q8 quant");
    EXPECT(il_gemv_q8_0(q8.data(), d.data(), x.data(), y.data(), N, K) == IL_OK, "q8 gemv");
    float rel8 = max_abs_diff(yref, y) / 16.f;
    EXPECT(rel8 < 0.05f, "q8 gemv error too large");
    std::vector<uint8_t> q4(N * K / 2);
    EXPECT(il_quantize_q4_0(W.data(), N * K, q4.data(), d.data()) == IL_OK, "q4 quant");
    EXPECT(il_gemv_q4_0(q4.data(), d.data(), x.data(), y.data(), N, K) == IL_OK, "q4 gemv");
    float rel4 = max_abs_diff(yref, y) / 16.f;
    EXPECT(rel4 < 0.3f, "q4 gemv error too large");
    EXPECT(il_quantize_q8_0(W.data(), 31, q8.data(), d.data()) == IL_ERR_SIZE, "q8 needs multiple of 32");
}

static void test_rope() {
    const int64_t T = 3, Hh = 2, hd = 8;
    auto x = randv(T * Hh * hd, 13);
    auto y = x;
    int64_t pos[3] = {0, 5, 100};
    EXPECT(il_rope(y.data(), T, Hh, hd, pos, 10000.f, 1) == IL_OK, "rope");
    // rotation preserves the norm of every pair
    for (int64_t i = 0; i < T * Hh * hd; i += 2) {
        float a = x[i] * x[i] + x[i + 1] * x[i + 1], b = y[i] * y[i] + y[i + 1] * y[i + 1];
        EXPECT(std::fabs(a - b) < 1e-4f, "rope norm");
    }
    for (int64_t i = 0; i < Hh * hd; ++i) EXPECT(std::fabs(x[i] - y[i]) < 1e-6f, "rope pos 0 is identity");
}

int main() {
    std::printf("%s simd=%s openmp=%d threads=%d\n", il_version(), il_simd_name(), il_has_openmp(),
                il_get_max_threads());
    test_matmul();
    test_errors();
    test_softmax_and_norm();
    test_attention();
    test_quant();
    test_rope();
    if (g_failures) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("all kernel tests passed\n");
    return 0;
}
