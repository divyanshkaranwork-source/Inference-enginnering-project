import { api } from "../api.js";
import { attachTip, bytes, clear, compact, fmt, h, ms } from "../dom.js";
import { hbars, legend, lines, roofline, segbar } from "../charts.js";
import {
  area, callout, card, field, message, num, page, runButton, select, slider, stats, table, text,
} from "../ui.js";

// ---------------------------------------------------------------------------
export const tokensPage = {
  id: "tokens",
  title: "Tokens & sampling",
  render(ctx) {
    // --- tokenizer ---
    const input = area("Hello world! Inference engineering makes LLMs fast 🚀 </s>", 3);
    const tokOut = h("div");
    const tokenize = async () => {
      const r = await api("/lab/tokenize", { text: input.value });
      tokOut.replaceChildren(
        stats([["Tokens", r.n_tokens, "incl. BOS"], ["Characters", r.n_chars],
          ["Chars / token", fmt(r.chars_per_token, 2), "~4 for English with Llama-2"], ["Vocabulary", r.vocab_size]]),
        h("div", { class: "tokens" }, r.tokens.map((t) =>
          attachTip(h("span", { class: "tok" }, t.piece, h("span", { class: "id", text: t.id })),
            `id ${t.id}\npiece ${JSON.stringify(t.piece)}`))));
    };
    let timer = null;
    input.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(() => tokenize().catch((e) => tokOut.replaceChildren(message("error", e.message))), 250);
    });
    tokenize().catch((e) => tokOut.replaceChildren(message("error", e.message)));

    // --- sampling ---
    const prompt = text(ctx.overview.model.name.startsWith("stories")
      ? "Once upon a time, there was a little" : "Once upon a time");
    const temp = slider("Temperature", 1.0, 0, 2, 0.05, (v) => v.toFixed(2));
    const topK = slider("Top-k (0 = off)", 0, 0, 50, 1);
    const topP = slider("Top-p", 1.0, 0.05, 1, 0.05, (v) => v.toFixed(2));
    const minP = slider("Min-p", 0, 0, 0.5, 0.01, (v) => v.toFixed(2));
    const sampOut = h("div");
    const run = async () => {
      const r = await api("/lab/sampling", {
        prompt: prompt.value, temperature: Number(temp.input.value), top_k: Number(topK.input.value),
        top_p: Number(topP.input.value), min_p: Number(minP.input.value),
      });
      sampOut.replaceChildren(
        stats([["Tokens kept", r.kept_tokens, "after filtering"], ["Entropy", `${fmt(r.entropy_nats, 2)} nats`, "raw distribution"],
          ["Top token", JSON.stringify(r.pieces[0]), `${(r.raw_probs[0] * 100).toFixed(1)}% raw`]]),
        legend([{ name: "raw softmax(logits)", cls: "c1" }, { name: "after temperature / top-k / top-p / min-p", cls: "c2" }]),
        hbars({
          categories: r.pieces.map((p) => JSON.stringify(p)),
          series: [{ name: "raw", cls: "c1", values: r.raw_probs }, { name: "processed", cls: "c2", values: r.processed_probs }],
          valueFmt: (v) => `${(v * 100).toFixed(1)}%`,
        }));
    };
    let st = null;
    for (const sl of [temp, topK, topP, minP]) {
      sl.input.addEventListener("change", () => { clearTimeout(st); st = setTimeout(() => run().catch(showErr), 50); });
    }
    const showErr = (e) => sampOut.replaceChildren(message("error", e.message));

    return page("Layer 1 · Foundations", "Tokens & sampling",
      "An LLM never sees text: the tokenizer maps text to integer ids, the transformer maps ids to a score (logit) " +
      "for every vocabulary entry, and the sampler turns scores into the next id. The loaded tokenizer is the " +
      "Llama-2 SentencePiece vocabulary (when a TinyStories model is downloaded) - the same one Llama-2-7B uses.",
      h("div", { class: "grid cols-2" },
        card("Tokenizer explorer", "Byte-fallback BPE. Note: typing </s> does NOT produce the end-of-sequence " +
          "control token - special-token injection is blocked.", field("Text", input), tokOut),
        card("Next-token distribution", "One forward pass over the prompt, then the sampling pipeline: " +
          "penalties → temperature → top-k → top-p → min-p → sample.",
          field("Prompt", prompt),
          h("div", { class: "form" }, temp.el, topK.el, topP.el, minP.el),
          h("div", { class: "row" }, runButton("Run forward pass", sampOut, run)),
          sampOut)),
      h("div", { class: "grid cols-3", style: { "margin-top": "16px" } },
        callout("Temperature", "divides logits by T. T→0 = greedy argmax; T>1 flattens the distribution."),
        callout("Top-k / top-p", "truncate the tail: keep the k best, or the smallest set whose mass ≥ p."),
        callout("Cost", "sampling is O(vocab) per sequence per step - tiny next to the forward pass, but engines still " +
          "fuse it on-GPU to avoid copying 32k-128k logits to the CPU.")));
  },
};

// ---------------------------------------------------------------------------
export const memoryPage = {
  id: "memory",
  title: "Memory & roofline",
  render(ctx) {
    const o = ctx.overview;
    const models = Object.keys(o.reference_models);
    const gpus = Object.keys(o.gpus);
    const dtypes = Object.keys(o.dtypes);
    const model = select(models, "llama-2-7b");
    const wdt = select(dtypes, "fp16");
    const kdt = select(dtypes, "fp16");
    const batch = num(16, 1, 4096);
    const seq = num(4096, 1, 1048576);
    const gpu = select(gpus, "A100-SXM-80GB");
    const nG = num(1, 1, 1024);
    const memOut = h("div");
    const runMem = async () => {
      const r = await api("/lab/memory", { model: model.value, weight_dtype: wdt.value, kv_dtype: kdt.value,
        batch: Number(batch.value), seq_len: Number(seq.value), gpu: gpu.value, n_gpus: Number(nG.value) });
      const gb = (v) => `${fmt(v, 1)} GB`;
      memOut.replaceChildren(
        r.fits ? message("ok", `Fits: ${gb(r.total_gb)} of ${gb(r.budget_gb)} usable (90% of ${r.n_gpus}× ${r.gpu}).`)
          : message("error", `Does not fit: needs ${gb(r.total_gb)} but only ${gb(r.budget_gb)} is usable. ` +
            "Quantize, shorten the context, lower the batch, or shard across GPUs (tensor parallelism)."),
        legend([{ name: "weights", cls: "c1" }, { name: "KV cache", cls: "c2" }, { name: "activations", cls: "c3" },
          { name: "free", cls: "cn" }]),
        segbar([{ name: "weights", value: r.weights_gb, cls: "c1" }, { name: "KV cache", value: r.kv_cache_gb, cls: "c2" },
          { name: "activations", value: r.activations_gb, cls: "c3" }], r.budget_gb, gb),
        stats([["Parameters", `${fmt(r.params_billion, 2)} B`], ["Weights", gb(r.weights_gb)],
          ["KV cache / token", `${fmt(r.kv_per_token_kb, 0)} KiB`, "2 × layers × kv_heads × head_dim × bytes"],
          ["KV / sequence", gb(r.kv_per_sequence_gb), `at ${seq.value} tokens`],
          ["Max KV tokens", compact(r.max_kv_tokens), "after weights + activations"],
          ["Max concurrent seqs", r.max_concurrent_seqs_at_seq_len, `at ${seq.value} tokens each`]]));
    };
    const rfModel = select(models, "llama-2-7b");
    const rfGpu = select(gpus, "H100-SXM-80GB");
    const rfDt = select(["fp16", "fp8", "int8", "q4_0"], "fp16");
    const rfOut = h("div");
    const runRf = async () => {
      const r = await api("/lab/roofline", { model: rfModel.value, gpu: rfGpu.value, weight_dtype: rfDt.value });
      rfOut.replaceChildren(
        legend([{ name: "roofline min(peak, AI × bandwidth)", cls: "c7" }, { name: "decode steps", cls: "c1" },
          { name: "prefill", cls: "c2" }]),
        roofline({ peakTflops: r.peak_tflops, bandwidthGBs: r.bandwidth_gbs,
          points: r.points.map((p) => ({ label: p.label, ai: p.arithmetic_intensity, tflops: p.attainable_tflops,
            phase: p.phase, bound: p.bound })) }),
        table([{ key: "label", label: "Operation" }, { key: "arithmetic_intensity", label: "FLOP/byte", num: true, fmt: (v) => fmt(v, 1) },
          { key: "bound", label: "Bound" },
          { key: "time_s", label: "Step time", num: true, fmt: (v) => ms(v * 1e3) },
          { key: "tokens_per_s", label: "Tokens/s", num: true, fmt: (v) => compact(v) }], r.points));
    };
    runMem().catch((e) => memOut.replaceChildren(message("error", e.message)));
    runRf().catch((e) => rfOut.replaceChildren(message("error", e.message)));
    for (const el of [model, wdt, kdt, batch, seq, gpu, nG]) el.addEventListener("change", () => runMem().catch((e) => memOut.replaceChildren(message("error", e.message))));
    for (const el of [rfModel, rfGpu, rfDt]) el.addEventListener("change", () => runRf().catch((e) => rfOut.replaceChildren(message("error", e.message))));

    return page("Layer 1 · Foundations", "Memory & roofline",
      "Two questions decide every serving design: does it fit in GPU memory, and is each step limited by compute " +
      "or by memory bandwidth? Llama-2-7B in fp16 is 13.5 GB of weights, and every decode step must read all of " +
      "them - plus the KV cache of every sequence.",
      card("GPU memory calculator", "weights + KV cache + activation workspace vs. 90% of GPU memory (vLLM's default " +
        "gpu_memory_utilization).",
        h("div", { class: "form" }, field("Model", model), field("Weight dtype", wdt), field("KV-cache dtype", kdt),
          field("Batch (sequences)", batch), field("Tokens / sequence", seq), field("GPU", gpu), field("GPUs", nG)),
        memOut),
      h("div", { style: { height: "16px" } }),
      card("Roofline", "Decode at batch 1 has arithmetic intensity ≈ 1 FLOP/byte; an H100 needs ≈ 300 to be compute-" +
        "bound. That gap is why batching gives (almost) free throughput until the ridge point.",
        h("div", { class: "form" }, field("Model", rfModel), field("GPU", rfGpu), field("Weights", rfDt)),
        rfOut));
  },
};

// ---------------------------------------------------------------------------
export const kvquantPage = {
  id: "kvquant",
  title: "KV cache & quantization",
  render(ctx) {
    const prompt = text("Once upon a time, there was a little robot who");
    const n = num(48, 4, 96);
    const kvOut = h("div");
    const runKV = async () => {
      kvOut.replaceChildren(message("info", "Generating twice (with and without the cache)…"));
      const r = await api("/lab/kv-cache", { prompt: prompt.value, new_tokens: Number(n.value) });
      const a = r.with_cache.decode_latencies_ms;
      const b = r.without_cache.decode_latencies_ms;
      kvOut.replaceChildren(
        stats([["With KV cache", `${fmt(r.with_cache.tokens_per_s, 0)} tok/s`, `TPOT ${ms(r.with_cache.tpot_ms)}`],
          ["Without", `${fmt(r.without_cache.tokens_per_s, 0)} tok/s`, `TPOT ${ms(r.without_cache.tpot_ms)}`],
          ["Speedup", `${fmt(r.without_cache.total_ms / r.with_cache.total_ms, 1)}×`, r.same_output ? "identical output ✓" : "outputs differ"],
          ["KV bytes / token", bytes(r.kv_bytes_per_token), `Llama-2-7B fp16: ${bytes(r.llama2_7b_kv_bytes_per_token_fp16)}`]]),
        legend([{ name: "without KV cache (recompute everything)", cls: "c2" }, { name: "with KV cache", cls: "c1" }]),
        lines({ series: [{ name: "without cache", cls: "c2", points: b.map((v, i) => [i + 2, v]) },
          { name: "with cache", cls: "c1", points: a.map((v, i) => [i + 2, v]) }],
        xLabel: "token #", yFmt: (v) => fmt(v, 1) + " ms", height: 220 }));
    };

    const kinds = ["fp32", "fp16", "bf16", "int8", "q8_0", "q4_0"];
    const qOut = h("div");
    const runQ = async () => {
      qOut.replaceChildren(message("info", "Re-quantizing every linear layer and re-running the model…"));
      const r = await api("/lab/quantization", { kinds });
      qOut.replaceChildren(
        hbars({ categories: r.results.map((x) => x.kind), series: [{ name: "weights", cls: "c1", values: r.results.map((x) => x.weights_mb) }],
          valueFmt: (v) => `${fmt(v, 1)} MiB` }),
        table([{ key: "kind", label: "Format" }, { key: "bits_per_weight", label: "bits/weight", num: true },
          { key: "weights_mb", label: "MiB", num: true, fmt: (v) => fmt(v, 1) },
          { key: "kl_divergence", label: "KL vs fp32", num: true, fmt: (v) => v.toExponential(2) },
          { key: "top1_agreement", label: "top-1 agree", num: true, fmt: (v) => `${(v * 100).toFixed(1)}%` },
          { key: "sample", label: "Greedy sample" }], r.results));
    };

    return page("Layer 1 · Foundations", "KV cache & quantization",
      "Attention needs the keys and values of every previous token. Recomputing them makes token n cost O(n); " +
      "caching them makes every decode step O(1) in compute - but the cache becomes the biggest memory consumer. " +
      "Quantization shrinks the other big consumer: the weights every step must read.",
      h("div", { class: "grid cols-2" },
        card("Experiment: KV cache on vs off", "Runs on the live model in this server.",
          h("div", { class: "form" }, field("Prompt", prompt), field("New tokens", n)),
          h("div", { class: "row" }, runButton("Run experiment", kvOut, runKV)), kvOut),
        card("Experiment: weight quantization", "Same prompt through fp32, fp16, bf16, int8 (per-channel), " +
          "Q8_0 and Q4_0 (llama.cpp blocks of 32). KL divergence measures how much the next-token distribution moved.",
          h("div", { class: "row" }, runButton("Quantize & compare", qOut, runQ)), qOut)),
      h("div", { class: "grid cols-3", style: { "margin-top": "16px" } },
        callout("GQA", "Llama-2-70B / Llama-3 share each K/V head across 8 query heads: 8× less KV cache."),
        callout("KV quantization", "fp8 / int8 KV caches halve memory again (vLLM --kv-cache-dtype fp8)."),
        callout("AWQ / GPTQ", "smarter 4-bit methods use calibration data to protect the weights that matter most.")));
  },
};
