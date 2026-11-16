import { api } from "../api.js";
import { bytes, fmt, h, ms } from "../dom.js";
import { columns, gantt, hbars, legend, segbar } from "../charts.js";
import { callout, card, check, field, message, num, page, runButton, select, seg, stats, table } from "../ui.js";

const STRATEGIES = {
  tp: { name: "Tensor (TP)", what: "Splits every weight matrix across GPUs (Megatron): column-parallel QKV/W1/W3, " +
    "row-parallel O/W2, 2 all-reduces per layer. Cuts weights + KV per GPU and per-token latency. Needs NVLink." },
  sp: { name: "Sequence (SP)", what: "Megatron-SP: the residual stream is sharded along the sequence; each all-reduce " +
    "becomes reduce-scatter + all-gather (same bytes) so norms and activations are split too." },
  pp: { name: "Pipeline (PP)", what: "Splits the layers into stages. Only activations cross stage boundaries " +
    "(cheap, works across nodes) but micro-batches are needed to fill the pipeline; bubbles waste time." },
  cp_ring: { name: "Context (ring)", what: "Ring Attention: each GPU keeps its query chunk while K/V chunks travel " +
    "around a ring; partial softmax results merge exactly. For very long prompts. Zig-zag fixes causal imbalance." },
  cp_ulysses: { name: "Context (Ulysses)", what: "DeepSpeed-Ulysses: all-to-all switches sharding from sequence to " +
    "heads, run normal attention per head group, all-to-all back." },
  ep: { name: "Expert (EP)", what: "MoE experts live on different GPUs; tokens are dispatched and combined with two " +
    "all-to-alls per layer. Load balance and capacity factor matter." },
  dp: { name: "Data (DP)", what: "Full replicas behind a router. Zero model communication; the routing policy " +
    "decides load balance and prefix-cache hit rate." },
};

function commTable(comm) {
  const rows = Object.entries(comm.by_op || {}).map(([op, d]) => ({ op, ...d }));
  return table([{ key: "op", label: "Collective" }, { key: "count", label: "Calls", num: true },
    { key: "bytes_per_rank", label: "Bytes / rank", num: true, fmt: (v) => bytes(v) },
    { key: "est_time_us", label: "Est. time", num: true, fmt: (v) => ms(v / 1000) }], rows);
}

export default {
  id: "parallel",
  title: "5D parallelism",
  render(ctx) {
    const o = ctx.overview;
    // ---------------- planner ----------------
    const model = select(Object.keys(o.reference_models), "llama-2-70b");
    const gpu = select(Object.keys(o.gpus), "H100-SXM-80GB");
    const dims = Object.fromEntries(["dp", "tp", "pp", "cp", "ep"].map((d) => [d, num(d === "tp" ? 8 : 1, 1, 64)]));
    const batch = num(32, 1, 4096);
    const ctxLen = num(4096, 128, 1048576);
    const wdt = select(["fp16", "fp8", "int8", "q4_0"], "fp16");
    const planOut = h("div");
    const runPlan = async () => {
      const r = await api("/lab/parallel/plan", { model: model.value, gpu: gpu.value, weight_dtype: wdt.value,
        batch: Number(batch.value), context_len: Number(ctxLen.value),
        ...Object.fromEntries(Object.entries(dims).map(([k, v]) => [k, Number(v.value)])) });
      if (!r.valid) {
        planOut.replaceChildren(...r.errors.map((e) => message("error", e)));
        return;
      }
      const m = r.memory_per_gpu_gb;
      const gb = (v) => `${fmt(v, 1)} GB`;
      planOut.replaceChildren(
        r.fits ? message("ok", `Fits on ${r.plan.world} × ${r.gpu}: ${gb(m.total)} of ${gb(m.budget)} per GPU.`)
          : message("error", `Does not fit: ${gb(m.total)} per GPU > ${gb(m.budget)}.`),
        ...r.warnings.map((w) => message("info", w)),
        legend([{ name: "weights", cls: "c1" }, { name: "KV cache", cls: "c2" }, { name: "activations", cls: "c3" },
          { name: "free", cls: "cn" }]),
        segbar([{ name: "weights", value: m.weights, cls: "c1" }, { name: "KV cache", value: m.kv_cache, cls: "c2" },
          { name: "activations", value: m.activations, cls: "c3" }], m.budget, gb),
        stats([["Latency / token", ms(r.decode.token_latency_ms), `${fmt(r.decode.tokens_per_s_per_user, 1)} tok/s per user`],
          ["Throughput", `${fmt(r.decode.tokens_per_s_total, 0)} tok/s`, `${r.plan.dp} replica(s)`],
          ["TP comm / stage", ms(r.decode.tp_comm_ms_per_stage)], ["Compute|memory / stage", ms(r.decode.compute_or_memory_ms_per_stage)],
          ["Max batch / replica", r.max_batch_per_replica_at_context, `at ${ctxLen.value} tokens`]]));
    };
    const nGpus = num(8, 1, 512);
    const recOut = h("div");
    const runRec = async () => {
      const r = await api("/lab/parallel/recommend", { model: model.value, gpu: gpu.value, n_gpus: Number(nGpus.value),
        batch: Number(batch.value), context_len: Number(ctxLen.value), weight_dtype: wdt.value });
      recOut.replaceChildren(r.length ? table([
        { key: (x) => `dp${x.plan.dp} × tp${x.plan.tp} × pp${x.plan.pp}` + (x.plan.ep > 1 ? ` · ep${x.plan.ep}` : ""), label: "Layout" },
        { key: (x) => x.decode.tokens_per_s_total, label: "Throughput (tok/s)", num: true, fmt: (v) => fmt(v, 0) },
        { key: (x) => x.decode.token_latency_ms, label: "Latency / token", num: true, fmt: ms },
        { key: (x) => x.memory_per_gpu_gb.total, label: "GB / GPU", num: true, fmt: (v) => fmt(v, 1) }], r)
        : message("error", "No layout fits - add GPUs, quantize, or reduce batch/context."));
    };
    for (const el of [model, gpu, batch, ctxLen, wdt, ...Object.values(dims)]) {
      el.addEventListener("change", () => runPlan().catch((e) => planOut.replaceChildren(message("error", e.message))));
    }
    runPlan().catch((e) => planOut.replaceChildren(message("error", e.message)));

    // ---------------- simulator ----------------
    let strat = "tp";
    const degree = num(2, 1, 8);
    const micro = num(4, 1, 16);
    const zig = check("zig-zag partitioning", false);
    const cap = num("", 0.1, 8, 0.1);
    cap.placeholder = "none";
    const skew = num(0, 0, 5, 0.5);
    const policy = select([["prefix_aware", "prefix-aware"], ["round_robin", "round-robin"], ["least_loaded", "least-loaded"]], "prefix_aware");
    const opts = h("div", { class: "form" });
    const desc = h("div");
    const simOut = h("div");
    const setStrat = (st) => {
      strat = st;
      desc.replaceChildren(callout(STRATEGIES[st].name, STRATEGIES[st].what));
      const extra = { pp: [field("Micro-batches", micro)], cp_ring: [h("div", { class: "field" }, zig.el)],
        ep: [field("Capacity factor", cap), field("Router skew", skew)], dp: [field("Routing policy", policy)] }[st] || [];
      opts.replaceChildren(field(st === "dp" ? "Replicas" : "Degree (GPUs)", degree), ...extra);
      simOut.replaceChildren();
    };
    const tabs = seg(Object.entries(STRATEGIES).map(([k, v]) => [k, v.name]), strat, setStrat);
    setStrat(strat);
    const runSim = async () => {
      const body = { strategy: strat, degree: Number(degree.value), microbatches: Number(micro.value),
        zigzag: zig.input.checked, skew: Number(skew.value), dp_policy: policy.value };
      if (cap.value !== "") body.capacity_factor = Number(cap.value);
      const r = await api("/lab/parallel/simulate", body);
      const parts = [];
      if (r.max_abs_error !== undefined) {
        parts.push(message(r.max_abs_error < 1e-3 ? "ok" : "error",
          `Max |difference| vs single-GPU result: ${r.max_abs_error.toExponential(2)}` +
          (r.max_abs_error < 1e-3 ? " - numerically identical." : " - outputs changed (e.g. dropped tokens).")));
      }
      if (strat === "tp" || strat === "sp") {
        parts.push(stats([["Weights / GPU", `${fmt(r.per_rank_weight_mb, 2)} MiB`, `of ${fmt(r.full_weight_mb, 2)} MiB`],
          ["Collectives", r.comm.num_collectives], ["Bytes / rank", bytes(r.comm.total_bytes_per_rank)]]));
      }
      if (strat === "pp") {
        const s = r.schedule;
        parts.push(stats([["Bubble", `${(s.bubble_fraction * 100).toFixed(0)}%`, `ideal (p-1)/(m+p-1) = ${(s.ideal_bubble_fraction * 100).toFixed(0)}%`],
          ["Makespan", ms(s.makespan_ms)], ["Stages", s.stages.map((x) => `L${x[0]}-${x[1]}`).join(" | ")]]),
        gantt({ rows: s.stages.length, rowLabel: (i) => `stage ${i}`,
          events: s.events.map((e) => ({ row: e.stage, start: e.start_ms, end: e.end_ms, label: `m${e.microbatch}`, cls: "c1",
            tip: `stage ${e.stage}, micro-batch ${e.microbatch}\n${fmt(e.start_ms, 2)}-${fmt(e.end_ms, 2)} ms` })) }),
        h("p", { class: "small muted", text: "Gaps are pipeline bubbles: stages waiting for their first or after their last micro-batch." }));
      }
      if (strat === "cp_ring") {
        parts.push(stats([["Load imbalance", `${fmt(r.load_imbalance, 2)}×`, "max / mean work"], ["Partition", r.scheme]]),
          h("p", { class: "small secondary", text: "Attention score entries computed per GPU" }),
          columns({ labels: r.work_per_rank.map((_, i) => `GPU ${i}`), series: [{ name: "work", cls: "c1", values: r.work_per_rank }], height: 160 }));
      }
      if (strat === "ep") {
        parts.push(stats([["Load imbalance", `${fmt(r.load_imbalance, 2)}×`, "busiest GPU vs mean"],
          ["Capacity / expert", r.capacity ?? "∞"], ["Dropped assignments", r.dropped_assignments]]),
        h("p", { class: "small secondary", text: "Tokens routed to each expert (top-2 of 8)" }),
        columns({ labels: r.tokens_per_expert.map((_, i) => `E${i}`), series: [{ name: "tokens", cls: "c3", values: r.tokens_per_expert }], height: 160 }));
      }
      if (strat === "dp") {
        parts.push(stats([["Prefix-cache hits", `${r.prefix_hit_tokens} tokens`, r.policy], ["Makespan", `${r.makespan_steps} steps`],
          ["Load imbalance", `${fmt(r.load_imbalance, 2)}×`]]),
        hbars({ categories: r.replicas.map((x) => `replica ${x.replica}`),
          series: [{ name: "requests", cls: "c1", values: r.replicas.map((x) => x.requests) }], valueFmt: (v) => fmt(v, 0) }),
        table([{ key: "replica", label: "Replica" }, { key: "requests", label: "Requests", num: true },
          { key: "steps", label: "Steps", num: true }, { key: "prefix_hit_tokens", label: "Prefix hits", num: true }], r.replicas));
      }
      if (r.comm) parts.push(h("h3", { text: "Communication" }), commTable(r.comm));
      simOut.replaceChildren(...parts);
    };

    return page("Scale-out", "5D parallelism",
      "When a model or its traffic outgrows one GPU you split it along five axes: data (replicas), tensor (matrices), " +
      "pipeline (layers), sequence/context (tokens) and expert (MoE experts). The planner applies the cost model to real " +
      "models and GPUs; the simulator runs each strategy on virtual GPUs and proves it computes the same result.",
      card("Planner", "Memory per GPU and decode speed for a layout. world = dp × tp × pp × cp.",
        h("div", { class: "form" }, field("Model", model), field("GPU", gpu), field("Weights", wdt), field("Batch / replica", batch),
          field("Context tokens", ctxLen), ...Object.entries(dims).map(([k, v]) => field(k.toUpperCase(), v))),
        planOut,
        h("div", { class: "row" }, field("GPUs available", nGpus), runButton("Recommend layouts", recOut, runRec, "")),
        recOut),
      h("div", { style: { height: "16px" } }),
      card("Simulator", "Runs on a tiny Llama / MoE layer so it finishes in milliseconds - the communication pattern is the real one.",
        tabs.el, desc, opts, h("div", { class: "row" }, runButton("Simulate", simOut, runSim)), simOut),
      h("div", { class: "grid cols-3", style: { "margin-top": "16px" } },
        callout("Rule 1", "TP inside a node (NVLink), PP or DP across nodes."),
        callout("Rule 2", "Decode is memory-bound: TP lowers latency, PP mostly adds capacity."),
        callout("Rule 3", "MoE: EP for experts, DP for attention (DeepSeek / SGLang DP-attention).")));
  },
};
