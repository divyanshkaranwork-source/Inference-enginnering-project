# Layer 4 — Production: serving it safely, observably, at scale

> Code: [`inferlab/server/`](../inferlab/server), [`deploy/`](../deploy) · Lab 11 · UI:
> *Playground*, *Production*

A fast engine is not yet a service. Production adds an API contract, streaming, admission control,
security, observability, deployment and a choice of engine for each piece of hardware.

## 1. The API contract: OpenAI-compatible

[`api.py`](../inferlab/server/api.py) implements `/v1/models`, `/v1/completions` and
`/v1/chat/completions`, with and without streaming. Every engine in this course (llama.cpp, vLLM,
SGLang, TensorRT-LLM) speaks the same protocol, which is why one gateway can front any of them and
why the OpenAI SDK, LangChain and friends work unchanged.

**Streaming** uses Server-Sent Events: one `data: {json}\n\n` per token and `data: [DONE]` at the
end. The user sees the first token after the TTFT instead of waiting for the full answer. If an error
happens after the stream has started, it is reported in-band as `data: {"error": ...}`, because the
HTTP status was already sent.

Extensions that are compatible with vLLM: `top_k`, `min_p`, `repetition_penalty`, `guided_choice`,
`cache_salt`, `ignore_eos`, and token-id prompts.

## 2. Latency SLOs and what drives them

| Metric | Definition | Driven by | Levers |
|--------|-----------|-----------|--------|
| **TTFT** | request arrival → first token | queueing + prefill | admission policy, chunked prefill, prefix caching, more replicas, prefill/decode disaggregation |
| **TPOT / ITL** | time between output tokens | decode step time (≈ weight + KV bytes / bandwidth) + interference | quantization, TP, smaller batch cap, speculative decoding |
| **E2E** | TTFT + TPOT × (n − 1) | both | — |
| **Throughput** | output tokens/s per GPU | batch size | continuous batching, KV memory (paging, quantized KV, GQA) |
| **Goodput** | throughput of requests that *meet* the SLOs | everything | the metric that matters for capacity planning |

Always look at **p90/p99**, never just averages. [`metrics.py`](../inferlab/server/metrics.py)
exports Prometheus histograms (`inferlab_ttft_seconds`, `inferlab_tpot_seconds`,
`inferlab_e2e_latency_seconds`), counters, and live engine gauges (running, waiting, KV usage,
prefix hit rate, preemptions). The *Production* page shows rolling percentiles.

The trade-off at the heart of serving: **bigger batches raise throughput but also raise TPOT**,
because each step does more work. Set `max_num_seqs` and `max_num_batched_tokens` from your TPOT SLO,
then add replicas to meet demand.

## 3. Admission control and backpressure

- per-key **token bucket** rate limit, returning `429` with `Retry-After`; CPU-heavy lab endpoints
  cost 5 tokens
- hard limits on body size (enforced while streaming the body, not just via `Content-Length`), on
  prompt length and on `max_tokens`
- a bounded in-flight queue, returning `503` when full. Rejecting fast is better than timing out
  slowly.
- a per-request deadline, after which the request is aborted and its blocks freed
- client disconnect also aborts the request, so abandoned generations don't burn the GPU

## 4. Security

LLM servers attract abuse because GPU time is expensive and prompts are sensitive.
[SECURITY.md](../SECURITY.md) lists the threat model and every control. These are the ones to
remember:

1. **Authenticate everything except health checks.** Keys are stored as hashes, compared in constant
   time, and never logged.
2. **Validate every input** with explicit bounds, and reject unknown fields on internal endpoints.
3. **Never log prompts or headers** by default; prompts are user data.
4. **Isolate tenants** in shared caches (`cache_salt`) and never use a non-cryptographic hash for
   cache keys.
5. **Treat model files as untrusted input**: no pickle, validate headers, and pin checksums.
6. **Harden the runtime**: non-root, read-only filesystem, no capabilities, minimal image, pinned
   dependencies and actions.
7. **Prevent SSRF** in gateways: upstream URLs come from configuration only, and redirects are not
   followed.
8. **Lock down the browser**: a strict CSP, no `innerHTML`, no inline scripts, frame denial.

## 5. Deployment ([`Dockerfile`](../Dockerfile), [`docker-compose.yml`](../docker-compose.yml), [`deploy/k8s`](../deploy/k8s/inferlab.yaml))

- multi-stage build: the compiler stays in the build stage, and kernel tests run during the build
- runs as UID 10001 with a read-only root filesystem, `cap_drop: ALL`, `no-new-privileges` and a
  tmpfs `/tmp`
- secrets are mounted as files (`INFERLAB_API_KEYS_FILE`), never baked into images or passed on
  command lines
- probes: `/healthz` (liveness, no dependencies) and `/readyz` (model loaded, backend healthy)
- **autoscaling**: CPU or GPU utilisation is a poor signal for LLM servers, because a saturated GPU
  can still be meeting its SLOs. Scale on **queue depth** (`inferlab_num_waiting_seqs`), KV usage or
  TTFT p90 (KEDA or prometheus-adapter).

## 6. Choosing an engine

See [06-engines.md](06-engines.md) for a detailed comparison and Llama-2-7B recipes. In short:
**llama.cpp** for CPUs, Apple Silicon, edge and quantized single-user use; **vLLM** as the general
GPU default; **SGLang** for prefix-heavy, agentic and structured-output workloads; **TensorRT-LLM**
for the last 20–40% on NVIDIA. Put InferLab-style gateway functions (auth, limits, metrics, routing)
in front of whichever you choose.

## 7. Beyond one replica

- **Routing**: replicas behind a router. Prefix-aware routing sends requests that share a prefix to
  the same replica, so its cache hits (5D page → DP simulation). Production routers include the
  SGLang router, NVIDIA Dynamo and llm-d.
- **Disaggregated prefill/decode** (DistServe, Splitwise, Mooncake): separate GPU pools for prefill
  (compute-bound) and decode (memory-bound), with the KV cache transferred between them. TTFT and
  TPOT can then be tuned independently.
- **KV offload / tiered caches**: spill cold prefix blocks to CPU RAM or NVMe (LMCache, Mooncake
  store) and bring them back instead of recomputing.
- **Multi-LoRA serving**: one base model, many adapters, batched together (Punica, S-LoRA, vLLM
  `--enable-lora`).

---

**Exercises**

1. Run `benchmarks/load_test.py` at concurrency 1, 4 and 16 and plot throughput against p99 TPOT.
   Where is the knee?
2. Add a per-key *token* quota (prompt + completion tokens per minute) next to the request rate limit.
3. Add an `/admin/drain` endpoint that stops admitting new requests and waits for in-flight ones to
   finish (graceful rolling deploys). Protect it with a separate admin key.
4. Put Caddy or nginx with TLS in front of `docker compose up`, enable `INFERLAB_HSTS`, and configure
   uvicorn's `--forwarded-allow-ips` for real client IPs.
