// Educational CUDA kernels for LLM inference + a self-checking test harness.
//
//   nvcc -O3 -arch=sm_80 -o kernels_test kernels.cu && ./kernels_test
//   (use -arch=sm_89 for RTX 40xx / L4, sm_90 for H100)
//
// These are the GPU versions of the CPU kernels in csrc/. They favour
// clarity over peak speed - production engines use cuBLAS/CUTLASS for GEMM,
// FlashAttention-2/3 or FlashInfer for attention, and hand-tuned PTX.
//
//   1. gemm_tiled          shared-memory tiling: each tile of A/B is loaded
//                          from HBM once and reused TILE times from SRAM
//   2. rmsnorm             one block per row, warp-shuffle reduction
//   3. flash_attn_fwd      FlashAttention: K/V tiles in shared memory,
//                          online softmax in registers, no T x T matrix
//   4. paged_attn_decode   vLLM-style decode attention through a block table

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(expr)                                                                       \
    do {                                                                                       \
        cudaError_t err__ = (expr);                                                            \
        if (err__ != cudaSuccess) {                                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, \
                         __LINE__);                                                            \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// 1. Tiled GEMM: C[M,N] = A[M,K] @ B[K,N]
// ---------------------------------------------------------------------------
template <int TILE>
__global__ void gemm_tiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                           int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE + 1];  // +1 padding avoids shared-memory bank conflicts
    const int tx = static_cast<int>(threadIdx.x), ty = static_cast<int>(threadIdx.y);
    const int row = static_cast<int>(blockIdx.y) * TILE + ty;
    const int col = static_cast<int>(blockIdx.x) * TILE + tx;
    float acc = 0.f;
    for (int t = 0; t < K; t += TILE) {
        As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.f;
        Bs[ty][tx] = (col < N && t + ty < K) ? B[(t + ty) * N + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// reductions
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// blockDim.x must be a multiple of 32 and <= 1024. All threads get the result.
__device__ float block_sum(float v) {
    __shared__ float partial[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    v = warp_sum(v);
    if (lane == 0) partial[wid] = v;
    __syncthreads();
    if (wid == 0) {
        float s = lane < nwarps ? partial[lane] : 0.f;
        s = warp_sum(s);
        if (lane == 0) total = s;
    }
    __syncthreads();
    return total;
}

__device__ float block_max(float v) {
    __shared__ float partial[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    v = warp_max(v);
    if (lane == 0) partial[wid] = v;
    __syncthreads();
    if (wid == 0) {
        float s = lane < nwarps ? partial[lane] : -INFINITY;
        s = warp_max(s);
        if (lane == 0) total = s;
    }
    __syncthreads();
    return total;
}

// ---------------------------------------------------------------------------
// 2. RMSNorm: one block per row
// ---------------------------------------------------------------------------
__global__ void rmsnorm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int dim,
                        float eps) {
    const float* xr = x + static_cast<size_t>(blockIdx.x) * dim;
    float* yr = y + static_cast<size_t>(blockIdx.x) * dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum(ss) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) yr[i] = xr[i] * inv * w[i];
}

// ---------------------------------------------------------------------------
// 3. FlashAttention forward (causal, single sequence). Q/K/V/O: [T, H, D].
//    grid = (ceil(T / BR), H), block = BR threads; thread i owns query row i.
// ---------------------------------------------------------------------------
template <int BR, int BC, int D>
__global__ void flash_attn_fwd(const float* __restrict__ Q, const float* __restrict__ K,
                               const float* __restrict__ V, float* __restrict__ O, int T, int H, float scale) {
    __shared__ float Ks[BC][D];
    __shared__ float Vs[BC][D];
    const int h = blockIdx.y;
    const int row0 = blockIdx.x * BR;
    const int i = row0 + threadIdx.x;
    float q[D], acc[D];
    float m = -INFINITY, l = 0.f;
    for (int d = 0; d < D; ++d) {
        q[d] = (i < T) ? Q[(static_cast<size_t>(i) * H + h) * D + d] * scale : 0.f;
        acc[d] = 0.f;
    }
    const int last_row = min(T - 1, row0 + BR - 1);  // causal: later keys are never needed
    for (int j0 = 0; j0 <= last_row; j0 += BC) {     // uniform across the block
        for (int idx = threadIdx.x; idx < BC * D; idx += BR) {
            const int r = idx / D, d = idx % D, j = j0 + r;
            Ks[r][d] = (j < T) ? K[(static_cast<size_t>(j) * H + h) * D + d] : 0.f;
            Vs[r][d] = (j < T) ? V[(static_cast<size_t>(j) * H + h) * D + d] : 0.f;
        }
        __syncthreads();
        if (i < T) {
            const int jmax = min(BC, i - j0 + 1);
            for (int r = 0; r < jmax; ++r) {
                float s = 0.f;
#pragma unroll
                for (int d = 0; d < D; ++d) s += q[d] * Ks[r][d];
                const float m_new = fmaxf(m, s);
                const float alpha = expf(m - m_new);  // exp(-inf) = 0 on the first key
                const float p = expf(s - m_new);
#pragma unroll
                for (int d = 0; d < D; ++d) acc[d] = acc[d] * alpha + p * Vs[r][d];
                l = l * alpha + p;
                m = m_new;
            }
        }
        __syncthreads();
    }
    if (i < T)
        for (int d = 0; d < D; ++d) O[(static_cast<size_t>(i) * H + h) * D + d] = acc[d] / l;
}

// ---------------------------------------------------------------------------
// 4. PagedAttention decode for one sequence. One block per query head.
//    k/v cache: [num_blocks, block_size, KVH, D]. Scores staged in shared mem.
// ---------------------------------------------------------------------------
template <int D>
__global__ void paged_attn_decode(const float* __restrict__ q, const float* __restrict__ kc,
                                  const float* __restrict__ vc, const int* __restrict__ block_table, int ctx,
                                  int block_size, int H, int KVH, float scale, float* __restrict__ out) {
    extern __shared__ float scores[];  // [ctx]
    const int h = blockIdx.x;
    const int kvh = h / (H / KVH);
    const float* qh = q + h * D;
    float local_max = -INFINITY;
    for (int t = threadIdx.x; t < ctx; t += blockDim.x) {
        const int blk = block_table[t / block_size];
        const float* k = kc + ((static_cast<size_t>(blk) * block_size + t % block_size) * KVH + kvh) * D;
        float s = 0.f;
#pragma unroll
        for (int d = 0; d < D; ++d) s += qh[d] * k[d];
        s *= scale;
        scores[t] = s;
        local_max = fmaxf(local_max, s);
    }
    const float mx = block_max(local_max);
    float local_sum = 0.f;
    for (int t = threadIdx.x; t < ctx; t += blockDim.x) {
        const float p = expf(scores[t] - mx);
        scores[t] = p;
        local_sum += p;
    }
    const float inv = 1.f / block_sum(local_sum);  // block_sum syncs, so scores[] is complete
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < ctx; ++t) {
            const int blk = block_table[t / block_size];
            acc += scores[t] * vc[((static_cast<size_t>(blk) * block_size + t % block_size) * KVH + kvh) * D + d];
        }
        out[h * D + d] = acc * inv;
    }
}

// ---------------------------------------------------------------------------
// test harness (CPU references)
// ---------------------------------------------------------------------------
static std::vector<float> randv(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

static float* to_dev(const std::vector<float>& h) {
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

static std::vector<float> to_host(const float* d, size_t n) {
    std::vector<float> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

static float max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
    return m;
}

static void cpu_attention(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
                          std::vector<float>& O, int Tq, int Tk, int H, int KVH, int D, int q_off) {
    const int g = H / KVH;
    const float scale = 1.f / std::sqrt(static_cast<float>(D));
    for (int h = 0; h < H; ++h)
        for (int i = 0; i < Tq; ++i) {
            const int lim = std::min(Tk, q_off + i + 1);
            std::vector<float> s(lim);
            float mx = -INFINITY, sum = 0.f;
            for (int j = 0; j < lim; ++j) {
                float acc = 0.f;
                for (int d = 0; d < D; ++d) acc += Q[(i * H + h) * D + d] * K[(j * KVH + h / g) * D + d];
                s[j] = acc * scale;
                mx = std::fmax(mx, s[j]);
            }
            for (int j = 0; j < lim; ++j) sum += (s[j] = std::exp(s[j] - mx));
            for (int d = 0; d < D; ++d) {
                float acc = 0.f;
                for (int j = 0; j < lim; ++j) acc += s[j] * V[(j * KVH + h / g) * D + d];
                O[(i * H + h) * D + d] = acc / sum;
            }
        }
}

int main() {
    int fails = 0;
    // GEMM
    {
        const int M = 123, N = 77, K = 200;
        auto A = randv(M * K, 1), B = randv(K * N, 2);
        std::vector<float> ref(M * N, 0.f);
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k)
                for (int j = 0; j < N; ++j) ref[i * N + j] += A[i * K + k] * B[k * N + j];
        float *dA = to_dev(A), *dB = to_dev(B), *dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
        dim3 block(16, 16), grid((N + 15) / 16, (M + 15) / 16);
        gemm_tiled<16><<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        const float err = max_diff(ref, to_host(dC, M * N));
        std::printf("gemm_tiled          max err %.2e\n", err);
        fails += err > 1e-3f;
        cudaFree(dA), cudaFree(dB), cudaFree(dC);
    }
    // RMSNorm
    {
        const int R = 8, Dm = 4096;
        auto x = randv(R * Dm, 3), w = randv(Dm, 4);
        std::vector<float> ref(R * Dm);
        for (int r = 0; r < R; ++r) {
            float ss = 0.f;
            for (int i = 0; i < Dm; ++i) ss += x[r * Dm + i] * x[r * Dm + i];
            const float inv = 1.f / std::sqrt(ss / Dm + 1e-5f);
            for (int i = 0; i < Dm; ++i) ref[r * Dm + i] = x[r * Dm + i] * inv * w[i];
        }
        float *dx = to_dev(x), *dw = to_dev(w), *dy = nullptr;
        CUDA_CHECK(cudaMalloc(&dy, R * Dm * sizeof(float)));
        rmsnorm<<<R, 256>>>(dx, dw, dy, Dm, 1e-5f);
        CUDA_CHECK(cudaGetLastError());
        const float err = max_diff(ref, to_host(dy, R * Dm));
        std::printf("rmsnorm             max err %.2e\n", err);
        fails += err > 1e-4f;
        cudaFree(dx), cudaFree(dw), cudaFree(dy);
    }
    // FlashAttention
    constexpr int D = 64;
    {
        const int T = 200, H = 4;
        auto Q = randv(T * H * D, 5), K = randv(T * H * D, 6), V = randv(T * H * D, 7);
        std::vector<float> ref(T * H * D);
        cpu_attention(Q, K, V, ref, T, T, H, H, D, 0);
        float *dQ = to_dev(Q), *dK = to_dev(K), *dV = to_dev(V), *dO = nullptr;
        CUDA_CHECK(cudaMalloc(&dO, T * H * D * sizeof(float)));
        constexpr int BR = 64, BC = 32;
        dim3 grid((T + BR - 1) / BR, H);
        flash_attn_fwd<BR, BC, D><<<grid, BR>>>(dQ, dK, dV, dO, T, H, 1.f / std::sqrt(float(D)));
        CUDA_CHECK(cudaGetLastError());
        const float err = max_diff(ref, to_host(dO, T * H * D));
        std::printf("flash_attn_fwd      max err %.2e\n", err);
        fails += err > 1e-3f;
        cudaFree(dQ), cudaFree(dK), cudaFree(dV), cudaFree(dO);
    }
    // PagedAttention decode
    {
        const int H = 8, KVH = 2, bs = 16, nblocks = 32, ctx = 150;
        auto K = randv(ctx * KVH * D, 8), V = randv(ctx * KVH * D, 9), q = randv(H * D, 10);
        std::vector<int> table = {5, 30, 2, 17, 9, 0, 31, 12, 3, 22};  // 10 blocks * 16 >= 150
        std::vector<float> kc(nblocks * bs * KVH * D, 0.f), vc(kc.size(), 0.f);
        for (int p = 0; p < ctx; ++p)
            for (int e = 0; e < KVH * D; ++e) {
                const size_t slot = (static_cast<size_t>(table[p / bs]) * bs + p % bs) * KVH * D + e;
                kc[slot] = K[p * KVH * D + e];
                vc[slot] = V[p * KVH * D + e];
            }
        std::vector<float> ref(H * D);
        cpu_attention(q, K, V, ref, 1, ctx, H, KVH, D, ctx - 1);
        float *dq = to_dev(q), *dk = to_dev(kc), *dv = to_dev(vc), *dout = nullptr;
        int* dt = nullptr;
        CUDA_CHECK(cudaMalloc(&dout, H * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dt, table.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(dt, table.data(), table.size() * sizeof(int), cudaMemcpyHostToDevice));
        paged_attn_decode<D><<<H, 128, ctx * sizeof(float)>>>(dq, dk, dv, dt, ctx, bs, H, KVH,
                                                              1.f / std::sqrt(float(D)), dout);
        CUDA_CHECK(cudaGetLastError());
        const float err = max_diff(ref, to_host(dout, H * D));
        std::printf("paged_attn_decode   max err %.2e\n", err);
        fails += err > 1e-4f;
        cudaFree(dq), cudaFree(dk), cudaFree(dv), cudaFree(dout), cudaFree(dt);
    }
    std::printf(fails ? "%d FAILED\n" : "all CUDA kernels passed\n", fails);
    return fails ? 1 : 0;
}
