# Glossary

| Term | Meaning |
|------|---------|
| **Arithmetic intensity (AI)** | FLOPs per byte moved. Below the ridge point a kernel is memory-bound; above it, compute-bound. |
| **AWQ / GPTQ** | 4-bit weight quantization methods that use calibration data to reduce error. |
| **Batch (continuous / static)** | Static: a fixed group runs to completion. Continuous: the batch is re-formed every step (Orca, vLLM). |
| **bf16** | 16-bit float with fp32's 8-bit exponent (wide range, 7-bit mantissa). |
| **Block / block table** | Fixed-size chunk of KV memory (e.g. 16 tokens) and the per-sequence map from logical to physical blocks (PagedAttention). |
| **Chunked prefill** | Splitting a long prompt across several steps so it can share steps with decode tokens. |
| **Context parallelism (CP)** | Splitting one sequence's tokens across GPUs (Ring Attention, Ulysses). |
| **Continuous batching** | See *Batch*. Also called iteration-level or in-flight batching. |
| **Data parallelism (DP)** | Independent model replicas serving different requests. |
| **Decode** | Generating one token per step for each running sequence. Memory-bound. |
| **Disaggregated serving** | Separate GPU pools for prefill and decode, with the KV cache transferred between them. |
| **E2E latency** | Request start to last token: TTFT + TPOT × (n − 1). |
| **EAGLE / Medusa / MTP** | Speculative decoding with lightweight draft heads on the target model. |
| **Expert parallelism (EP)** | Placing MoE experts on different GPUs, with all-to-all token routing. |
| **FlashAttention** | Tiled attention with online softmax that never materialises the T×T score matrix. |
| **FP8 / FP4** | 8-bit / 4-bit floating-point formats with hardware support on Hopper/Blackwell. |
| **GEMM / GEMV** | Matrix-matrix / matrix-vector multiply. Prefill ≈ GEMM; batch-1 decode ≈ GEMV. |
| **GGUF** | llama.cpp's model file format (weights + metadata + tokenizer, often quantized). |
| **Goodput** | Throughput counting only requests that met their latency SLOs. |
| **GQA / MQA / MHA** | Grouped / multi-query / multi-head attention: how many query heads share each K/V head. |
| **ITL** | Inter-token latency (≈ TPOT, often reported per token rather than averaged). |
| **KV cache** | Stored key/value vectors of previous tokens so each decode step only processes the new token. |
| **Logits / logprobs** | Raw scores per vocabulary token / their log-softmax. |
| **MoE** | Mixture of Experts: a router picks k of E feed-forward experts per token. |
| **Online softmax** | Single-pass max + normaliser with rescaling; the core of FlashAttention. |
| **PagedAttention** | KV cache in non-contiguous fixed-size blocks with block tables (vLLM). |
| **Pipeline parallelism (PP)** | Splitting layers into stages on different GPUs, with micro-batches flowing through. |
| **Preemption** | Evicting a running sequence's KV when memory runs out (recompute or swap later). |
| **Prefill** | Processing the prompt tokens in one pass. Compute-bound. Sets TTFT. |
| **Prefix caching** | Reusing the KV of shared prompt prefixes across requests (hashed blocks or a radix tree). |
| **Q4_0 / Q8_0 / Q4_K_M** | llama.cpp block quantization formats (32-value blocks with scales; K-quants use super-blocks). |
| **RadixAttention** | SGLang's radix-tree prefix cache over token sequences. |
| **Ridge point** | Peak FLOP/s ÷ memory bandwidth: the AI where a device becomes compute-bound. |
| **RMSNorm** | Normalisation by root-mean-square (no mean subtraction, no bias). |
| **RoPE** | Rotary position embeddings: rotate Q/K pairs by position-dependent angles. |
| **Roofline** | Attainable performance = min(peak compute, AI × bandwidth). |
| **Sequence parallelism (SP)** | Megatron-SP: sharding the norm/residual activations along the sequence alongside TP. |
| **SLO** | Service-level objective, e.g. p99 TTFT < 500 ms. |
| **Speculative decoding** | Drafting several tokens cheaply and verifying them in one target pass, with the output distribution unchanged. |
| **SSE** | Server-Sent Events: the HTTP streaming format used by the OpenAI API. |
| **SwiGLU** | Llama's FFN activation: silu(xW1) ⊙ (xW3), then W2. |
| **Tensor parallelism (TP)** | Splitting each weight matrix across GPUs (Megatron), with all-reduce after row-parallel layers. |
| **TPOT** | Time per output token during decode. |
| **TTFT** | Time to first token: queueing + prefill. |
