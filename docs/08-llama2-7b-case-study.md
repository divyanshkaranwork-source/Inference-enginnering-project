# Case study: every number for Llama-2-7B

The course's reference model, worked through end to end. Every value can be reproduced with
`inferlab info`, `inferlab plan` or the UI's *Memory & roofline* and *5D parallelism* pages.
Hardware numbers are approximate datasheet values; real systems reach 60–80% of them.

## Architecture

| | |
|---|---|
| layers | 32 |
| d_model | 4096 |
| heads / KV heads | 32 / 32 (multi-head attention, no GQA) |
| head_dim | 128 |
| FFN (SwiGLU) | 11008 |
| vocabulary | 32000 (SentencePiece BPE, the same as the TinyStories models) |
| context | 4096, RoPE θ = 10000 |
| parameters | **6,738,415,616** (attention 2.15 B, FFN 4.33 B, embeddings 0.13 B, LM head 0.13 B) |

## Memory

| Item | fp16 | int8 | 4-bit (Q4_K_M) |
|------|------|------|----------------|
| weights | **13.5 GB** | 6.9 GB | ~4.1 GB |
| KV per token | **512 KiB** | 256 KiB (int8 KV) | — |
| KV per 4096-token sequence | 2.0 GiB | 1.0 GiB | — |

On one A100-80GB at 90% utilisation, with fp16 weights and KV: 72 GB − 13.5 GB weights − ~0.4 GB
activations leaves about **58 GB for KV, or about 110k tokens**. That is about 27 concurrent
full-context sequences, or about 220 sequences of 512 tokens. This is exactly the number vLLM prints
as "# GPU blocks" (× 16 tokens per block).

## Compute

- FLOPs per generated token ≈ 2 × 6.6 B (non-embedding) ≈ **13.2 GFLOP**, plus attention
  `4 · L · d · context` (≈ 0.5 GFLOP at 1k context)
- prefill of a 2048-token prompt ≈ 2048 × 13.5 GFLOP ≈ 28 TFLOP → about 28 ms of pure compute on an
  H100 (989 TFLOPS), about 45 ms at realistic efficiency

## Decode speed (memory-bound)

The lower bound for one decode step is (weight bytes + KV bytes) / bandwidth:

| GPU | Bandwidth | fp16 batch 1 | fp16 batch 32 (1k ctx) | 4-bit batch 1 |
|-----|-----------|--------------|------------------------|---------------|
| RTX 4090 | 1.0 TB/s | ~13 ms → ~50–75 tok/s | ~1,000+ tok/s total | ~4 ms → 150+ tok/s |
| A100-80GB | 2.0 TB/s | ~7 ms → ~100–140 tok/s | ~1,500 tok/s total | — |
| H100-SXM | 3.35 TB/s | ~4 ms → ~170–250 tok/s | ~2,500 tok/s total | — |
| M2 Max | 0.4 TB/s | ~34 ms → ~20–30 tok/s | — | ~10 ms → ~60–90 tok/s |

Arithmetic intensity at batch 1 is about 1 FLOP/byte; the H100 ridge point is about 295. Decode stays
memory-bound until the batch is in the hundreds, which is why serving systems batch aggressively and
why quantization speeds up single-user decode.

## Picking a deployment

| Goal | Recommended |
|------|-------------|
| laptop / CPU / Mac | llama.cpp, Q4_K_M GGUF (~4 GB), `-np 2-4` |
| one 24 GB GPU (4090, L4, A10G) | vLLM or SGLang fp16 (13.5 GB weights + ~8 GB KV), or AWQ 4-bit for more concurrency |
| one 80 GB GPU, high throughput | vLLM/SGLang with prefix caching, `max-num-seqs` 64–256; FP8 on H100 |
| lowest latency per user | TensorRT-LLM FP8 + speculative decoding (EAGLE/draft); TP=2 halves the per-token bytes |
| many users | DP replicas (one GPU each; 7B never needs TP for capacity) + a prefix-aware router |

## Try it

```bash
inferlab info --model llama-2-7b --gpu A100-SXM-80GB --batch 32 --context 1024
inferlab plan --model llama-2-7b --gpu L4-24GB --gpus 4 --recommend
docker compose -f deploy/engines/vllm.yml up      # then compare with the numbers above
```
