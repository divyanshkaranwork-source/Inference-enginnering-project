# Production engines: llama.cpp, vLLM, SGLang, TensorRT-LLM

> Code: [`inferlab/runtime/backends/`](../inferlab/runtime/backends), [`deploy/engines/`](../deploy/engines)
> · UI: *Production → Engines compared*

Everything in Layers 1–3 is implemented, at full scale, by four open-source engines. Once you know
the concepts, each engine is a set of choices about them. InferLab's gateway can front any of them
(`INFERLAB_BACKEND=llamacpp|vllm|sglang|tensorrt_llm`), so you keep the same API, auth, limits and
metrics while swapping the engine underneath.

## Side by side

| | **llama.cpp** | **vLLM** | **SGLang** | **TensorRT-LLM** |
|---|---|---|---|---|
| Language / stack | C/C++ (ggml), no Python | Python + PyTorch + custom CUDA/Triton kernels | Python + PyTorch + FlashInfer/Triton kernels | C++ runtime, PyTorch backend, TensorRT engines |
| Hardware | CPU (AVX/NEON), Apple Metal, CUDA, ROCm, Vulkan, SYCL | NVIDIA, AMD, TPU, Intel, CPU | NVIDIA, AMD, TPU | NVIDIA only |
| Weights | GGUF, quantized (Q2_K…Q8_0, IQ) | HF safetensors; AWQ, GPTQ, FP8, INT8, bitsandbytes | HF; AWQ, GPTQ, FP8 | HF → optimised engine; FP8, FP4, INT8, INT4-AWQ |
| Batching | continuous over N slots (`-np`) | continuous + chunked prefill | continuous + chunked prefill, zero-overhead scheduler | in-flight batching |
| KV cache | per-slot ring, quantizable (q8_0/q4_0) | PagedAttention, hashed prefix cache | RadixAttention (token tree), cache-aware scheduling | paged KV, KV reuse, offload |
| Speculative | draft model, lookup / n-gram | draft, EAGLE-1/3, Medusa, MTP, n-gram | EAGLE-2/3, MTP | draft-target, Medusa, EAGLE, lookahead, MTP |
| Structured output | GBNF grammars | XGrammar / guidance | XGrammar, fast constrained decoding (compressed FSM) | XGrammar |
| Parallelism | layer / row split across GPUs | TP, PP, DP, EP | TP, DP-attention, EP (DeepEP), PP | TP, PP, EP, CP |
| Best at | local/edge/CPU, single user, small memory | general-purpose GPU serving, model coverage | prefix-heavy, agents, multi-turn, structured output | lowest latency / highest throughput on NVIDIA |
| Concept to study | quantized GEMV kernels | PagedAttention + scheduler | RadixAttention + frontend language | kernel fusion, compilation, FP8/FP4 |

## Mapping to InferLab code

| Concept | InferLab | llama.cpp | vLLM | SGLang | TensorRT-LLM |
|---------|----------|-----------|------|--------|--------------|
| Engine step loop | `runtime/engine.py` | `server.cpp` `update_slots` | `v1/engine/core.py` | `managers/scheduler.py` | `batch_manager` / `PyExecutor` |
| Scheduler | `runtime/scheduler.py` | slot assignment | `v1/core/sched/scheduler.py` | `schedule_policy.py` | capacity scheduler |
| KV memory | `runtime/block_manager.py` | `llama_kv_cache` | `v1/core/kv_cache_manager.py` | `mem_cache/radix_cache.py` | `kvCacheManager` |
| Attention kernel | `csrc/src/attention.cpp` | `ggml` flash-attn | FlashAttention / FlashInfer / Triton backends | FlashInfer / FA3 / Triton | XQA / MMHA / FMHA |
| Quantized matmul | `csrc/src/quant.cpp` | `ggml-quants.c`, `mmq`/`mmvq` | Marlin, Machete, CUTLASS FP8 | same family | FP8/FP4 GEMMs |
| OpenAI server | `server/api.py` | `llama-server` | `entrypoints/openai/` | `entrypoints/http_server.py` | `trtllm-serve` |

(Paths are indicative and change as the projects evolve; search the repositories for these names.)

## Llama-2-7B recipes

All four compose files in [`deploy/engines/`](../deploy/engines) start the engine on an internal
network and the InferLab gateway on `127.0.0.1:9000`.

### llama.cpp: 7B on a laptop

```bash
python convert_hf_to_gguf.py Llama-2-7b-chat-hf --outtype f16 --outfile llama-2-7b-chat.f16.gguf
llama-quantize llama-2-7b-chat.f16.gguf llama-2-7b-chat.Q4_K_M.gguf Q4_K_M   # 13.5 GB -> ~4 GB
llama-server -m llama-2-7b-chat.Q4_K_M.gguf -c 8192 -np 4 --host 127.0.0.1 --port 8080   # add -ngl 99 on a GPU
```

What to observe: Q4_K_M decode speed ≈ memory bandwidth / 4 GB. On an M2 Max (400 GB/s) that is
about 60–90 tok/s; on a DDR5 laptop (~60 GB/s), about 10–15 tok/s. This is the roofline of Layer 1
in practice.

### vLLM: the general-purpose GPU default

```bash
vllm serve meta-llama/Llama-2-7b-chat-hf --max-model-len 4096 --enable-prefix-caching \
     --gpu-memory-utilization 0.9 --max-num-seqs 64        # + --quantization fp8 --kv-cache-dtype fp8 on H100/L40S
```

What to observe in the startup log: "# GPU blocks: N". That is the PagedAttention pool (N × 16
tokens). Compare it with `inferlab info --model llama-2-7b --gpu <yours>`.

### SGLang: prefix-heavy and agentic workloads

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 \
     --context-length 4096 --mem-fraction-static 0.85
```

What to observe: send the same long system prompt with many different questions. The cached-token
counters in the logs climb (RadixAttention).

### TensorRT-LLM: maximum NVIDIA performance

```bash
trtllm-serve meta-llama/Llama-2-7b-chat-hf --max_batch_size 64 --max_seq_len 4096 --port 8000
```

What to observe: the lowest TTFT/TPOT of the four on the same GPU, especially with FP8 on Hopper.
The cost is NVIDIA lock-in and engine build and compile time.

## Benchmarking them fairly

```bash
export INFERLAB_API_KEY=$(cat secrets/api_keys.txt)
python benchmarks/load_test.py --url http://127.0.0.1:9000 --requests 200 --concurrency 32 --max-tokens 256
```

- use the same model, precision, `max_tokens`, prompt-length distribution and concurrency for every
  engine
- report **p50/p99 TTFT and TPOT plus throughput** at several concurrency levels; a single number
  hides the latency–throughput trade-off
- warm up first (CUDA graphs, autotuning, prefix caches)
- for maximum-throughput numbers, use each project's own benchmark tool as well (`vllm bench serve`,
  `python -m sglang.bench_serving`, `trtllm-bench`)
