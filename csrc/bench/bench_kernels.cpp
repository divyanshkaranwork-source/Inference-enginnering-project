// Standalone C++ benchmark:  ./build/bench_kernels [size]
// Prints GFLOP/s for the GEMM ladder and effective GB/s for GEMV variants.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include "inferlab_kernels.h"

using clk = std::chrono::steady_clock;

static double time_it(const std::function<void()>& fn, int reps) {
    fn();  // warm-up
    auto t0 = clk::now();
    for (int i = 0; i < reps; ++i) fn();
    return std::chrono::duration<double>(clk::now() - t0).count() / reps;
}

int main(int argc, char** argv) {
    long n_arg = argc > 1 ? std::strtol(argv[1], nullptr, 10) : 512;
    if (n_arg < 32 || n_arg > 4096) {
        std::fprintf(stderr, "size must be in [32, 4096]\n");
        return 2;
    }
    const int64_t n = n_arg;
    std::mt19937 rng(0);
    std::normal_distribution<float> d(0.f, 1.f);
    std::vector<float> A(n * n), B(n * n), C(n * n);
    for (auto& x : A) x = d(rng);
    for (auto& x : B) x = d(rng);
    std::printf("%s  simd=%s  openmp=%d  threads=%d\n\n", il_version(), il_simd_name(), il_has_openmp(),
                il_get_max_threads());
    std::printf("GEMM %ldx%ldx%ld\n", (long)n, (long)n, (long)n);
    const double flops = 2.0 * n * n * n;
    struct G {
        const char* name;
        il_status (*fn)(const float*, const float*, float*, int64_t, int64_t, int64_t);
        int reps;
    } gemms[] = {{"naive (i-j-k)", il_matmul_naive, 1},
                 {"reordered (i-k-j)", il_matmul_reordered, 3},
                 {"tiled", il_matmul_tiled, 3},
                 {"tiled + OpenMP", il_matmul_parallel, 5}};
    for (auto& g : gemms) {
        double t = time_it([&] { g.fn(A.data(), B.data(), C.data(), n, n, n); }, g.reps);
        std::printf("  %-20s %8.2f ms  %7.2f GFLOP/s\n", g.name, t * 1e3, flops / t / 1e9);
    }

    const int64_t N = 4096, K = 4096;  // one Llama-2-7B projection matrix
    std::vector<float> W(N * K), x(K), y(N);
    for (auto& v : W) v = d(rng);
    for (auto& v : x) v = d(rng);
    std::vector<int8_t> q8(N * K);
    std::vector<uint8_t> q4(N * K / 2);
    std::vector<float> s(N * K / IL_QK);
    std::printf("\nGEMV %ldx%ld (decode step of one 4096x4096 linear layer)\n", (long)N, (long)K);
    double t = time_it([&] { il_gemv(W.data(), x.data(), y.data(), N, K); }, 20);
    std::printf("  %-20s %8.3f ms  %7.2f GB/s\n", "fp32", t * 1e3, N * K * 4.0 / t / 1e9);
    il_quantize_q8_0(W.data(), N * K, q8.data(), s.data());
    t = time_it([&] { il_gemv_q8_0(q8.data(), s.data(), x.data(), y.data(), N, K); }, 20);
    std::printf("  %-20s %8.3f ms  %7.2f GB/s (of q8 data)\n", "q8_0", t * 1e3, N * K * 34.0 / 32 / t / 1e9);
    il_quantize_q4_0(W.data(), N * K, q4.data(), s.data());
    t = time_it([&] { il_gemv_q4_0(q4.data(), s.data(), x.data(), y.data(), N, K); }, 20);
    std::printf("  %-20s %8.3f ms  %7.2f GB/s (of q4 data)\n", "q4_0", t * 1e3, N * K * 18.0 / 32 / t / 1e9);
    return 0;
}
