# Advanced topics: where to go next

These are the topics working inference engineers deal with beyond the core of the course. Each entry
gives the core idea, how it connects to code in this repository, and what to read or try next.

## Precision and quantization

- **FP8 (E4M3/E5M2)** on Hopper, Ada and Blackwell: weights *and* activations in 8 bits with per-tensor
  or per-channel scales. It halves both bytes and FLOPs relative to fp16 and is the default fast path
  in TensorRT-LLM and vLLM on H100. **FP4 / NVFP4 / MXFP4** on Blackwell go further, with micro-block
  scales (16 or 32 values per scale): the same idea as llama.cpp's Q4_0 blocks, implemented in hardware.
- **AWQ** (activation-aware weight quantization) scales salient input channels before 4-bit rounding.
  **GPTQ** quantizes column by column and uses approximate second-order information to compensate the
  error. **SmoothQuant** moves activation outliers into the weights so W8A8 works.
- **KV-cache quantization** (fp8 or int8 KV): halves the cache, doubling concurrency or context.
  *Try:* add an int8 option to `PagedKVPool` with per-block scales.
- **Batch invariance / determinism**: results can change with batch composition because reductions
  are ordered differently. Deterministic kernels fix the reduction order per element.

## Attention variants

- **MQA → GQA → MLA**: DeepSeek's Multi-head Latent Attention caches a compressed latent vector instead
  of full K/V, giving a 5–10× smaller cache, with matrix absorption at decode time.
- **Sliding window / local attention** (Mistral, Gemma) bounds the KV per sequence. **Attention sinks /
  StreamingLLM** keep the first few tokens plus a window.
- **Long context**: RoPE scaling (linear, NTK, YaRN, Llama-3.1's scaling) extends the trained context.
  The cost is KV memory and prefill time, which is where context parallelism (Lab 10) and chunked
  prefill (Layer 2) come in.
- **Linear attention / SSMs** (Mamba, RWKV, hybrid models): constant-size state instead of a growing
  KV cache. Engines must manage two cache types.

## Runtime and systems

- **CUDA Graphs**: record the decode step's kernel launches once and replay them, removing CPU launch
  overhead (crucial at small batch). vLLM and SGLang capture graphs per batch size.
- **torch.compile / kernel fusion compilers**: fuse elementwise chains automatically; TensorRT does
  this as an ahead-of-time compiler.
- **Overlap scheduling**: prepare step N+1's batch on the CPU while the GPU runs step N (SGLang's
  zero-overhead scheduler).
- **Prefill/decode disaggregation** (DistServe, Splitwise, Mooncake, NVIDIA Dynamo, llm-d): separate
  pools tuned for TTFT and for TPOT, with the KV cache moved between them over NVLink/RDMA (NIXL).
- **Tiered KV caches**: GPU → CPU → SSD → remote (LMCache, Mooncake store) for multi-turn chat and agents.
- **Multi-LoRA serving**: batch requests for different adapters with gathered matmuls (Punica's SGMV,
  S-LoRA paging adapters).
- **Structured generation**: grammar-constrained decoding with precomputed token masks (XGrammar),
  plus jump-forward decoding of deterministic spans (SGLang).
- **Sampling on GPU**: top-k/top-p via sorting-free rejection sampling (FlashInfer) instead of
  sorting the vocabulary.

## Distributed inference

- **Wide EP for MoE** (DeepSeek-V3/R1): 256 experts over dozens of GPUs with DeepEP all-to-all
  kernels, redundant hot experts (EPLB) and two-batch overlap to hide communication.
- **Communication/computation overlap** for TP: split GEMMs so all-reduce chunks overlap compute
  (async TP, fused GEMM + reduce-scatter).
- **Speculative decoding at scale**: EAGLE-3 draft heads; tree attention to verify several
  candidate branches at once.

## Evaluating serving changes

- Never ship a quantization or kernel change without both a **quality** check (perplexity or task
  evals against the fp16 baseline, KL divergence as in Lab 05) and a **performance** check (p50/p99
  TTFT/TPOT at the target concurrency).
- Use realistic request distributions: prompt and output lengths, prefix sharing, and arrival
  process (Poisson, bursty). Averages over uniform synthetic prompts mislead.

## Suggested reading order

1. vLLM paper, *Efficient Memory Management for LLM Serving with PagedAttention* (SOSP'23)
2. Orca (OSDI'22) on iteration-level scheduling; Sarathi-Serve on chunked prefill
3. FlashAttention 1/2/3 papers
4. SGLang paper (RadixAttention); the DistServe paper (disaggregation)
5. Leviathan et al. / Chen et al. 2023 (speculative decoding); EAGLE
6. Megatron-LM (tensor parallelism), GPipe (pipeline), Ring Attention, DeepSpeed-Ulysses
7. AWQ, GPTQ, SmoothQuant; the FP8 formats paper
8. Then read the source: llama.cpp `ggml`, vLLM `vllm/v1/`, SGLang `srt/`, TensorRT-LLM `_torch/`
