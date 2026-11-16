import { api } from "../api.js";
import { bytes, fmt, h, ms } from "../dom.js";
import { callout, card, code, message, page, runButton, stats, table } from "../ui.js";

const COMPARE = [
  ["Best at", "laptops, CPUs, Apple Silicon, edge", "high-throughput GPU serving, widest model support",
    "prefix-heavy & structured workloads, agents", "max performance on NVIDIA GPUs"],
  ["Core idea", "GGUF quantized weights, hand-tuned SIMD/Metal/CUDA kernels",
    "PagedAttention + continuous batching", "RadixAttention + cache-aware scheduler, fast grammar decoding",
    "compiled engines, fused kernels, FP8/FP4, in-flight batching"],
  ["Batching", "slots (-np) with continuous batching", "continuous, chunked prefill", "continuous, chunked prefill",
    "in-flight batching"],
  ["KV cache", "per-slot contiguous, quantizable", "paged blocks, hashed prefix cache", "radix-tree token cache",
    "paged blocks, KV reuse"],
  ["Quantization", "Q2_K…Q8_0, IQ formats", "AWQ, GPTQ, FP8, INT8, Marlin kernels", "AWQ, GPTQ, FP8",
    "FP8, FP4, INT8 SmoothQuant, INT4 AWQ"],
  ["Parallelism", "tensor split / layer offload", "TP, PP, DP, EP", "TP, PP, DP-attention, EP", "TP, PP, EP, CP"],
  ["Llama-2-7B on one GPU", "~4 GB Q4_K_M, runs on 8 GB laptops", "13.5 GB fp16", "13.5 GB fp16", "13.5 GB fp16 / 7 GB fp8"],
];

let timer = null;

export default {
  id: "production",
  title: "Production",
  teardown() { clearInterval(timer); },
  render(ctx) {
    const o = ctx.overview;
    const slo = h("div");
    const engines = h("div");
    const refresh = async () => {
      const m = await api("/lab/metrics/summary");
      const row = (name, d) => [name, d ? ms(d.p50) : "–", d ? ms(d.p90) : "–", d ? ms(d.p99) : "–", d ? ms(d.mean) : "–"];
      slo.replaceChildren(
        stats([["Requests (window)", m.requests], ["Tokens generated", fmt(m.tokens_total, 0)]]),
        table([{ key: 0, label: "Metric" }, { key: 1, label: "p50", num: true }, { key: 2, label: "p90", num: true },
          { key: 3, label: "p99", num: true }, { key: 4, label: "mean", num: true }],
        [row("TTFT - time to first token", m.ttft_ms), row("TPOT - time per output token", m.tpot_ms),
          row("E2E - end-to-end latency", m.e2e_ms)]),
        m.requests ? null : message("info", "No requests yet - use the Playground or fire a burst on the Live engine page."));
    };
    const loadEngines = async () => {
      const r = await api("/lab/engines");
      const a = r.active;
      engines.replaceChildren(
        message(a.health && a.health.ok ? "ok" : "error",
          `Active backend: ${a.backend} (${a.model})${a.upstream ? " → " + a.upstream : ""} - ` +
          `${a.health && a.health.ok ? "healthy" : "unhealthy"}`),
        table([{ key: 0, label: "" }, { key: 1, label: "llama.cpp" }, { key: 2, label: "vLLM" }, { key: 3, label: "SGLang" },
          { key: 4, label: "TensorRT-LLM" }], COMPARE),
        h("h3", { text: "Serve Llama-2-7B with a real engine, keep this gateway in front" }),
        table([{ key: "title", label: "Engine" }, { key: "launch", label: "Launch command", fmt: (v) => h("code", { text: v }) },
          { key: "default_url", label: "Default URL" }], Object.values(r.presets)),
        code("# run an engine (see deploy/engines/*.yml), then point InferLab at it:\n" +
          "INFERLAB_BACKEND=vllm INFERLAB_UPSTREAM_URL=http://127.0.0.1:8000 \\\n" +
          "INFERLAB_UPSTREAM_MODEL=meta-llama/Llama-2-7b-chat-hf inferlab serve --port 9000"));
    };
    refresh().catch((e) => slo.replaceChildren(message("error", e.message)));
    loadEngines().catch((e) => engines.replaceChildren(message("error", e.message)));
    clearInterval(timer);
    timer = setInterval(() => { if (!document.hidden) refresh().catch(() => {}); }, 3000);

    const sec = o.security;
    const controls = [
      ["Authentication", sec.auth ? "API keys (SHA-256 digests, constant-time compare)" : "disabled (loopback only)", sec.auth],
      ["Rate limiting", `token bucket ${fmt(sec.rate_limit_rps, 1)} req/s, burst ${sec.rate_limit_burst} per key`, true],
      ["Request size", `body ≤ ${bytes(sec.max_body_bytes)}, prompt ≤ ${sec.max_prompt_chars} chars, max_tokens ≤ ${sec.max_tokens}`, true],
      ["Timeouts", `${sec.request_timeout_s}s per request; disconnects abort generation and free KV blocks`, true],
      ["Browser hardening", "strict CSP (no inline script), nosniff, frame-ancestors none, no referrer", true],
      ["CORS", sec.cors_origins.length ? sec.cors_origins.join(", ") : "off (same-origin only)", true],
      ["Metrics endpoint", sec.metrics_public ? "public" : "requires API key", !sec.metrics_public],
      ["Model files", "parsed as raw data (no pickle); downloads over HTTPS with SHA-256 checks", true],
      ["Prefix cache", "SHA-256 block hashes (CVE-2025-25183 class), optional per-tenant cache_salt", true],
    ];
    return page("Layer 4 · Production", "Production",
      "Serving is a product: latency SLOs, capacity, safety and the right engine for the hardware. This page shows the " +
      "gateway's live SLOs, how it fronts llama.cpp / vLLM / SGLang / TensorRT-LLM, and the security controls in force.",
      h("div", { class: "grid cols-2" },
        card("Live SLOs", "Rolling window of the last 500 requests. Watch p99 - tail latency is what users feel.",
          slo, h("div", { class: "row" }, runButton("Refresh", slo, refresh, ""))),
        card("Security controls", "Enforced by this server right now.",
          table([{ key: 0, label: "Control" }, { key: 1, label: "Setting" },
            { key: 2, label: "Status", fmt: (v) => h("span", { class: "pill" },
              h("span", { class: `dot ${v ? "good" : "warn"}` }), v ? "on" : "review") }], controls))),
      h("div", { style: { height: "16px" } }),
      card("Engines compared", null, engines),
      h("div", { class: "grid cols-3", style: { "margin-top": "16px" } },
        callout("Scale out", "replicas behind a cache-aware router (5D parallelism page → DP); autoscale on queue depth, not CPU."),
        callout("Disaggregation", "run prefill and decode on separate GPU pools (DistServe, Dynamo, llm-d) to meet TTFT and TPOT independently."),
        callout("Observe", "scrape /metrics with Prometheus: inferlab_ttft_seconds, inferlab_tpot_seconds, KV usage, queue depth.")));
  },
};
