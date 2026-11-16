import { bytes, compact, fmt, h } from "../dom.js";
import { callout, card, page, table } from "../ui.js";

const LAYERS = [
  {
    id: "tokens", name: "1 · Foundations", sub: "know what it costs", cls: "c4",
    chips: ["Llama architecture", "tokenizer", "sampling", "KV cache", "memory math", "roofline", "quantization"],
  },
  {
    id: "engine", name: "2 · Runtime", sub: "keep the GPU busy", cls: "c3",
    chips: ["continuous batching", "PagedAttention", "prefix caching", "chunked prefill", "preemption",
      "speculative decoding", "structured output"],
  },
  {
    id: "kernels", name: "3 · Kernels", sub: "make each op fast", cls: "c2",
    chips: ["tiled GEMM", "GEMV", "FlashAttention", "PagedAttention kernel", "online softmax", "Q8_0 / Q4_0",
      "fused RMSNorm", "CUDA + Triton"],
  },
  {
    id: "production", name: "4 · Production", sub: "serve it safely at scale", cls: "c1",
    chips: ["OpenAI-compatible API", "SSE streaming", "API-key auth", "rate limiting", "TTFT / TPOT SLOs",
      "Prometheus", "llama.cpp · vLLM · SGLang · TensorRT-LLM"],
  },
  {
    id: "parallel", name: "5D parallelism", sub: "when one GPU is not enough", cls: "c7",
    chips: ["data", "tensor (+sequence)", "pipeline", "context (ring / Ulysses)", "expert"],
  },
];

const FLOW = ["HTTP request", "auth + rate limit", "tokenize", "queue", "schedule (budget)", "prefill",
  "decode loop ×N", "sample", "detokenize", "stream SSE"];

export default {
  id: "overview",
  title: "Overview",
  render(ctx) {
    const o = ctx.overview;
    const m = o.model;
    const ref = o.reference_models["llama-2-7b"];
    const stack = h("div", { class: "stack" }, LAYERS.map((L) =>
      h("a", { class: "layer", href: `#/${L.id}` },
        h("span", { class: `bar ${L.cls}` }),
        h("span", { class: "name" }, L.name, h("small", { text: L.sub })),
        h("span", { class: "chips" }, L.chips.map((c) => h("span", { class: "chip", text: c }))))));

    const rows = [
      ["parameters", compact(m.n_params), compact(ref.n_params)],
      ["layers", m.n_layers, ref.n_layers],
      ["d_model", m.dim, ref.dim],
      ["query heads / KV heads", `${m.n_heads} / ${m.n_kv_heads}`, `${ref.n_heads} / ${ref.n_kv_heads}`],
      ["head dim", m.head_dim, ref.head_dim],
      ["FFN hidden (SwiGLU)", m.hidden_dim, ref.hidden_dim],
      ["vocabulary", m.vocab_size, ref.vocab_size],
      ["context length", m.max_seq_len, ref.max_seq_len],
      ["KV cache / token (fp16)", bytes(2 * m.n_layers * m.kv_dim * 2), bytes(2 * ref.n_layers * ref.kv_dim * 2)],
      ["weights (fp16)", bytes(m.n_params * 2), bytes(ref.n_params * 2)],
    ];
    const tiny = m.name === "tiny-random";
    return page("Inference engineering lab", "How LLMs are delivered fast",
      "A complete serving stack you can read and run: a Llama-2 engine in NumPy with C++ kernels, a vLLM-style " +
      "runtime, an OpenAI-compatible production gateway, and a 5D-parallelism simulator. Every number on these " +
      "pages is computed live by the server you are connected to.",
      tiny ? callout("Running a random-weight model.",
        "Text will look like noise. Run `inferlab download` (fetches TinyStories Llama-2 15M + the Llama-2 tokenizer) " +
        "and restart the server to generate real stories. Every mechanism works the same way either way.") : null,
      h("div", { class: "grid cols-2", style: { "margin-top": "16px" } },
        card("The four layers", "Learn them in order - each layer builds on the previous one. Click a layer to open its lab.", stack),
        card("Live model vs. the course reference model",
          "Same Llama-2 architecture - every lesson on the small model transfers to Llama-2-7B.",
          table([{ key: 0, label: "" }, { key: 1, label: m.name, num: true }, { key: 2, label: "Llama-2-7B", num: true }],
            rows),
          h("p", { class: "small muted",
            text: `Loaded weights: ${bytes(o.model_bytes)} (${o.linear_kind}) · kernels: ${o.kernels.selected}` +
              (o.kernels.available ? ` (C++ ${o.kernels.simd}, ${o.kernels.threads} threads)` : " (C++ library not built)") }))),
      h("div", { class: "section-title" }, h("h2", { text: "Life of a request" })),
      card("From HTTP to tokens", "What happens to one request in this server (and in vLLM / SGLang).",
        h("div", { class: "flow" }, FLOW.flatMap((st, i) => [
          i ? h("span", { class: "arrow", text: "→" }) : null, h("span", { class: "step", text: st })])),
        h("div", { class: "grid cols-3" },
          callout("Prefill", "processes the whole prompt in one pass - compute-bound, sets time-to-first-token."),
          callout("Decode", "one token per step for every running sequence - memory-bandwidth-bound."),
          callout("Batching", "many sequences share each weight read - the single biggest throughput lever.")),
      ),
      h("div", { class: "section-title" }, h("h2", { text: "Suggested path" })),
      card("Learning path", "Read the matching chapter in docs/ alongside each page.",
        table([{ key: 0, label: "#" }, { key: 1, label: "Page" }, { key: 2, label: "You will learn" },
          { key: 3, label: "Doc" }], [
          [1, "Tokens & sampling", "tokenizer, logits, temperature / top-k / top-p", "docs/01-foundations.md"],
          [2, "Memory & roofline", "why decode is memory-bound; KV-cache sizing", "docs/01-foundations.md"],
          [3, "KV cache & quantization", "O(n²) → O(n); int8 / Q4 trade-offs", "docs/01-foundations.md"],
          [4, "Live engine", "PagedAttention, continuous batching, preemption", "docs/02-runtime.md"],
          [5, "Batching & prefix cache", "static vs continuous; RadixAttention", "docs/02-runtime.md"],
          [6, "Speculative decoding", "draft + verify, acceptance rate", "docs/02-runtime.md"],
          [7, "Kernels", "tiling, SIMD, FlashAttention, quantized GEMV", "docs/03-kernels.md"],
          [8, "Production", "SLOs, gateways, engines, security", "docs/04-production.md"],
          [9, "5D parallelism", "DP / TP / PP / CP / EP and when to use which", "docs/05-parallelism.md"],
        ])),
      h("p", { class: "small muted", style: { "margin-top": "12px" },
        text: `Security: auth ${o.security.auth ? "on" : "OFF (loopback only)"}, ` +
          `rate limit ${fmt(o.security.rate_limit_rps, 1)} req/s (burst ${o.security.rate_limit_burst}), ` +
          `body ≤ ${bytes(o.security.max_body_bytes)}, max_tokens ≤ ${o.security.max_tokens}.` }));
  },
};
