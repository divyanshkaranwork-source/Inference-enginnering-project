# GPU kernels (CUDA + Triton)

The CPU kernels in [`csrc/`](../csrc) teach the *algorithms*. The kernels here show how the same
algorithms map onto GPU hardware. They need an NVIDIA GPU, so CI does not build them. Both files
check themselves against a reference implementation.

| File | Kernels | Build / run |
|------|---------|-------------|
| `cuda/kernels.cu` | tiled GEMM (shared memory), RMSNorm (warp shuffles), FlashAttention forward, PagedAttention decode | `nvcc -O3 -arch=sm_80 -o kernels_test cuda/kernels.cu && ./kernels_test` |
| `triton/kernels.py` | fused softmax, RMSNorm, causal FlashAttention (tensor cores via `tl.dot`) | `pip install torch triton && python triton/kernels.py` |

Use `-arch=sm_89` for Ada (RTX 40xx, L4, L40S), `sm_90` for H100/H200 and `sm_100` for Blackwell.

## CPU → GPU mental model

| CPU concept (csrc/) | GPU equivalent |
|---------------------|----------------|
| cache tile (L1/L2) | shared memory (SRAM, ~228 KB per SM on H100) |
| SIMD lane (AVX2: 8 floats) | thread in a warp (32 threads in lockstep) |
| `#pragma omp parallel for` over tiles | grid of thread blocks |
| horizontal sum (`hsum`) | `__shfl_xor_sync` warp reduction |
| FMA instruction | tensor-core MMA (`mma.sync`, `wgmma` on Hopper) |
| DRAM bandwidth (~50-300 GB/s) | HBM bandwidth (2-8 TB/s) |

## Where the real kernels live

- **FlashAttention 2/3**: `Dao-AILab/flash-attention` (CUTLASS/CuTe; FA3 uses Hopper TMA + warp specialization)
- **FlashInfer**: paged/ragged attention kernels used by SGLang and vLLM
- **vLLM `csrc/`**: `paged_attention_v1/v2`, fused RMSNorm, rotary embedding, quantized GEMMs (Marlin, Machete)
- **TensorRT-LLM**: XQA/MMHA decode attention, FP8/FP4 GEMMs, fused MoE
- **llama.cpp `ggml-cuda/`**: quantized matmul (`mmq`), `mmvq` for decode GEMV, flash attention for every GGUF type
