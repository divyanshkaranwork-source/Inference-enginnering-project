import { api } from "../api.js";
import { attachTip, fmt, h, ms } from "../dom.js";
import { columns, legend, lines } from "../charts.js";
import { area, callout, card, check, field, message, num, page, runButton, select, slider, stats, table } from "../ui.js";

// ---------------------------------------------------------------------------
let pollTimer = null;

export const enginePage = {
  id: "engine",
  title: "Live engine",
  teardown() {
    clearInterval(pollTimer);
    pollTimer = null;
  },
  render(ctx) {
    const isNative = ctx.overview.backend.backend === "native";
    const top = h("div");
    const grid = h("div", { class: "kv-grid", role: "img", "aria-label": "KV cache blocks" });
    const gridStats = h("div");
    const running = h("div", { class: "seq-list" });
    const waiting = h("div", { class: "small muted" });
    const hist = h("div");
    const queues = h("div");
    const finished = h("div");
    const err = h("div");
    let cells = [];

    const draw = (st) => {
      const bs = st.block_stats;
      top.replaceChildren(stats([
        ["Engine step", fmt(st.step, 0)], ["Running", st.running.length, `max ${st.config.max_num_seqs}`],
        ["Waiting", st.num_waiting], ["KV blocks used", `${bs.num_blocks - bs.free_blocks} / ${bs.num_blocks}`,
          `${(bs.usage * 100).toFixed(0)}% · ${bs.block_size} tokens/block`],
        ["Prefix cache hit rate", `${(bs.prefix_hit_rate * 100).toFixed(1)}%`, `${bs.prefix_hits_tokens} tokens reused`],
        ["Generated tokens", fmt(st.totals.generation_tokens, 0), `${st.totals.preemptions} preemptions`]]));
      if (cells.length !== st.blocks.length) {
        grid.replaceChildren();
        cells = st.blocks.map(() => {
          const c = h("div", { class: "kv-cell" });
          attachTip(c, "");
          grid.appendChild(c);
          return c;
        });
      }
      st.blocks.forEach((b, i) => {
        const c = cells[i];
        c.className = `kv-cell ${b.state === "free" ? "" : b.state}`;
        c.dataset.tip = `block ${b.id}: ${b.state}` + (b.owners.length ? `\nowned by ${b.owners.join(", ")}` : "") +
          (b.ref > 1 ? `\nref count ${b.ref} (shared prefix)` : "");
      });
      gridStats.textContent = `${bs.cached_blocks} blocks hold reusable prefix content · ${bs.evictions} evictions`;
      running.replaceChildren(...(st.running.length ? st.running.map((s) => {
        const total = s.prompt_tokens + s.output_tokens;
        const cached = Math.min(s.cached_prompt_tokens, total);
        return h("div", { class: "seq" },
          h("span", { class: "mono", text: s.id }),
          attachTip(h("div", { class: "bar" },
            h("span", { class: "cachedpart", style: { width: (cached / total) * 100 + "%" } }),
            h("span", { class: "done", style: { width: ((s.computed_tokens - cached) / total) * 100 + "%" } })),
          `${s.id} (${s.phase})\nprompt ${s.prompt_tokens} + output ${s.output_tokens} tokens\n` +
          `computed ${s.computed_tokens}, from prefix cache ${s.cached_prompt_tokens}\nblocks [${s.blocks.join(", ")}]`));
      }) : [h("span", { class: "muted small", text: "idle - fire a burst to see continuous batching" })]));
      waiting.textContent = st.num_waiting ? `waiting: ${st.waiting.map((w) => w.id).join(", ")}` : "";
      const H = st.history.slice(-60);
      if (H.length) {
        hist.replaceChildren(
          legend([{ name: "prefill tokens", cls: "c2" }, { name: "decode tokens", cls: "c1" }]),
          columns({ labels: H.map((x) => x.step), series: [
            { name: "prefill", cls: "c2", values: H.map((x) => x.prefill_tokens) },
            { name: "decode", cls: "c1", values: H.map((x) => x.decode_tokens) }], height: 170 }));
        queues.replaceChildren(
          legend([{ name: "running", cls: "c1" }, { name: "waiting", cls: "c2" }]),
          lines({ series: [
            { name: "running", cls: "c1", points: H.map((x) => [x.step, x.num_running]) },
            { name: "waiting", cls: "c2", points: H.map((x) => [x.step, x.num_waiting]) }],
          xLabel: "step", yFmt: (v) => fmt(v, 0), height: 150 }));
      }
      finished.replaceChildren(table([{ key: "id", label: "Request" }, { key: "tokens", label: "Tokens", num: true },
        { key: "ttft_ms", label: "TTFT", num: true, fmt: ms }, { key: "tpot_ms", label: "TPOT", num: true, fmt: ms },
        { key: "cached_prompt_tokens", label: "Prefix hit", num: true }, { key: "reason", label: "Finish" }],
      [...st.recent_finished].reverse()));
    };

    const poll = async () => {
      try {
        draw(await api("/lab/engine/state"));
        err.replaceChildren();
      } catch (e) {
        err.replaceChildren(message("error", e.message));
      }
    };
    const n = num(8, 1, 16);
    const mt = num(96, 1, 128);
    const shared = check("shared system prompt (prefix caching)", true);
    const burst = runButton("Fire burst", err, async () => {
      await api("/lab/engine/burst", { n: Number(n.value), max_tokens: Number(mt.value), shared_prefix: shared.input.checked });
      poll();
    });
    if (isNative) {
      poll();
      pollTimer = setInterval(() => { if (!document.hidden) poll(); }, 500);
    }
    return page("Layer 2 · Runtime", "Live engine",
      "This is the real scheduler and paged KV cache serving every request to this server, updated twice a second. " +
      "Fire a burst of concurrent requests and watch sequences join and leave the batch at every step (continuous " +
      "batching), blocks being allocated on demand (PagedAttention) and shared between requests with the same " +
      "prefix (prefix caching).",
      isNative ? null : message("info", "Live engine state needs INFERLAB_BACKEND=native."),
      card("Generate load", null, h("div", { class: "form" }, field("Requests", n), field("Max tokens each", mt),
        h("div", { class: "field" }, shared.el)), h("div", { class: "row" }, burst), err),
      h("div", { style: { height: "16px" } }),
      top,
      h("div", { class: "grid cols-2", style: { "margin-top": "16px" } },
        card("KV-cache blocks (PagedAttention)", "Each square is one block of KV memory. Hover for owners.",
          legend([{ name: "used", cls: "c1" }, { name: "shared by several sequences", cls: "c2" },
            { name: "cached (free but reusable)", cls: "c3" }, { name: "free", cls: "cn" }]),
          grid, h("p", { class: "small muted" }, gridStats)),
        card("Running batch", "Bar = tokens computed so far (aqua = served from the prefix cache).", running, waiting)),
      h("div", { class: "grid cols-2", style: { "margin-top": "16px" } },
        card("Tokens processed per step", "Chunked prefill mixes prompt chunks with decode tokens under the " +
          "max_num_batched_tokens budget.", hist),
        card("Queue depth per step", null, queues)),
      h("div", { style: { height: "16px" } }),
      card("Recently finished", null, finished));
  },
};

// ---------------------------------------------------------------------------
export const batchingPage = {
  id: "batching",
  title: "Batching & prefix cache",
  render() {
    const nReq = num(8, 2, 16);
    const maxT = num(32, 4, 64);
    const seqs = num(4, 1, 16);
    const bOut = h("div");
    const runB = async () => {
      bOut.replaceChildren(message("info", "Running the same workload through two engines…"));
      const r = await api("/lab/batching-compare", { num_requests: Number(nReq.value), max_tokens: Number(maxT.value),
        max_num_seqs: Number(seqs.value) });
      const S = r.static;
      const C = r.continuous;
      bOut.replaceChildren(
        table([{ key: 0, label: "" }, { key: 1, label: "Static", num: true }, { key: 2, label: "Continuous", num: true }], [
          ["engine steps", S.steps, C.steps],
          ["throughput", `${fmt(S.tokens_per_s, 0)} tok/s`, `${fmt(C.tokens_per_s, 0)} tok/s`],
          ["mean TTFT", ms(S.mean_ttft_ms), ms(C.mean_ttft_ms)],
          ["mean end-to-end", ms(S.mean_e2e_ms), ms(C.mean_e2e_ms)]]),
        legend([{ name: "static batching", cls: "c2" }, { name: "continuous batching", cls: "c1" }]),
        lines({ series: [
          { name: "static", cls: "c2", points: S.batch_size_per_step.map((v, i) => [i + 1, v]) },
          { name: "continuous", cls: "c1", points: C.batch_size_per_step.map((v, i) => [i + 1, v]) }],
        xLabel: "step", yFmt: (v) => fmt(v, 0), height: 200 }),
        h("p", { class: "small muted", text: "Sequences in the batch at each step. Static batching waits for the " +
          "longest request before admitting new ones, so the batch drains and the GPU idles." }));
    };

    const prompts = area([
      "You are a helpful assistant. Answer briefly. What is a GPU?",
      "You are a helpful assistant. Answer briefly. What is a TPU?",
      "You are a helpful assistant. Explain the KV cache.",
      "Translate to French: good morning",
    ].join("\n"), 5);
    const rOut = h("div");
    const runR = async () => {
      const list = prompts.value.split("\n").map((x) => x.trim()).filter(Boolean).slice(0, 16);
      const r = await api("/lab/radix", { prompts: list });
      const render = (node) => h("li", null,
        node.tokens ? h("span", { class: "node" }, h("span", { class: "t", text: JSON.stringify(node.text) }),
          h("span", { class: "muted small", text: `${node.tokens} tok` })) : h("span", { class: "node", text: "root" }),
        node.children.length ? h("ul", null, node.children.map(render)) : null);
      rOut.replaceChildren(
        stats([["Tokens stored", r.total_tokens], ["Hit rate", `${(r.hit_rate * 100).toFixed(1)}%`, "prompt tokens already cached"]]),
        h("ul", { class: "tree" }, render(r.tree)),
        table([{ key: (row) => list[r.inserts.indexOf(row)], label: "Prompt" },
          { key: "tokens", label: "Tokens", num: true }, { key: "cached", label: "Reused", num: true }], r.inserts));
    };

    return page("Layer 2 · Runtime", "Batching & prefix caching",
      "Iteration-level (continuous) batching was introduced by Orca and popularised by vLLM; every modern engine " +
      "uses it. Prefix caching avoids recomputing shared prompt prefixes - vLLM hashes full KV blocks, SGLang keeps " +
      "a radix tree of token sequences (RadixAttention).",
      h("div", { class: "grid cols-2" },
        card("Static vs continuous batching", "Requests have different output lengths (4…max tokens).",
          h("div", { class: "form" }, field("Requests", nReq), field("Max tokens", maxT), field("max_num_seqs", seqs)),
          h("div", { class: "row" }, runButton("Compare", bOut, runB)), bOut),
        card("RadixAttention tree (SGLang)", "One prompt per line. Shared prefixes become shared tree edges.",
          field("Prompts", prompts), h("div", { class: "row" }, runButton("Build tree", rOut, runR)), rOut)),
      h("div", { class: "grid cols-3", style: { "margin-top": "16px" } },
        callout("Chunked prefill", "long prompts are split so decode steps of other users never stall behind them."),
        callout("Preemption", "when KV blocks run out, the newest sequence is evicted and recomputed later."),
        callout("Cache isolation", "prefix hashes are SHA-256 and can be salted per tenant (cache_salt) so users " +
          "cannot probe each other's prompts through timing.")));
  },
};

// ---------------------------------------------------------------------------
export const specPage = {
  id: "speculative",
  title: "Speculative decoding",
  render(ctx) {
    const prompt = area("Once upon a time, there was a little girl named Lily. Lily liked to play outside. " +
      "One day, Lily went outside to play.", 3);
    const method = select([["ngram", "n-gram prompt lookup"], ["draft_model", "draft model (first N layers)"]], "ngram");
    const k = slider("Draft length k", 4, 1, 8, 1);
    const layers = num(Math.max(1, Math.floor(ctx.overview.model.n_layers / 2)), 1, ctx.overview.model.n_layers);
    const maxT = num(64, 1, 128);
    const temp = slider("Temperature", 0, 0, 1.5, 0.05, (v) => v.toFixed(2));
    const out = h("div");
    const run = async () => {
      out.replaceChildren(message("info", "Running baseline and speculative decoding…"));
      const r = await api("/lab/speculative", { prompt: prompt.value, method: method.value, k: Number(k.input.value),
        max_tokens: Number(maxT.value), draft_layers: Number(layers.value), temperature: Number(temp.input.value) });
      const sp = r.speculative;
      out.replaceChildren(
        stats([["Acceptance rate", `${(sp.acceptance_rate * 100).toFixed(0)}%`, `${sp.accepted}/${sp.proposed} drafts`],
          ["Tokens per target pass", fmt(sp.tokens_per_target_forward, 2), "1.0 = no gain"],
          ["Target forward passes", sp.target_forwards, `baseline: ${r.baseline.completion_tokens}`],
          ["Wall-clock speedup", r.speedup ? `${fmt(r.speedup, 2)}×` : "–", "CPU numbers; GPUs gain more"],
          ["Output identical", r.identical_output === null ? "n/a (sampling)" : r.identical_output ? "yes ✓" : "no",
            "greedy must match exactly"]]),
        legend([{ name: "accepted draft", cls: "cgood" }, { name: "rejected draft", cls: "ccrit" },
          { name: "token from the target model", cls: "c1" }]),
        h("div", { class: "spec-rounds" }, sp.trace.map((rd, i) => h("div", { class: "spec-round" },
          h("span", { class: "n", text: `round ${i + 1}` }),
          h("span", { class: "tokens" },
            rd.proposed_pieces.map((p, j) => h("span", { class: `tok ${j < rd.accepted ? "accepted" : "rejected"}`, text: p })),
            h("span", { class: "tok bonus", text: rd.emitted_pieces[rd.emitted_pieces.length - 1] }))))),
        h("h3", { text: "Generated text" }), h("div", { class: "output", text: sp.text }));
    };
    return page("Layer 2 · Runtime", "Speculative decoding",
      "A cheap drafter guesses k tokens; the target model checks all of them in ONE forward pass. Because decode is " +
      "memory-bound, verifying k+1 tokens costs about the same as generating 1. Rejection sampling keeps the output " +
      "distribution exactly the target's.",
      card("Experiment", "n-gram lookup needs no extra model and shines when the output repeats the input " +
        "(summaries, code edits). A draft model works everywhere but costs its own forward passes.",
        field("Prompt", prompt),
        h("div", { class: "form" }, field("Drafter", method), k.el, field("Draft layers", layers),
          field("Max tokens", maxT), temp.el),
        h("div", { class: "row" }, runButton("Run", out, run)), out),
      h("div", { class: "grid cols-3", style: { "margin-top": "16px" } },
        callout("EAGLE / Medusa", "draft heads on the target's own hidden states: high acceptance, no second model."),
        callout("MTP", "DeepSeek-V3 trains multi-token-prediction heads that double as drafters."),
        callout("When it hurts", "large batches are already compute-bound: speculation then wastes FLOPs.")));
  },
};
