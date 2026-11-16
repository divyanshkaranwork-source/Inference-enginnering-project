import { api } from "../api.js";
import { fmt, h, ms } from "../dom.js";
import { hbars } from "../charts.js";
import { callout, card, code, field, message, num, page, runButton, select, seg, table } from "../ui.js";

const OPS = {
  matmul: {
    title: "GEMM: C = A·B",
    why: "Prefill and large-batch decode are matrix-matrix products. The ladder shows what each optimisation buys: " +
      "loop order (unit-stride access → SIMD), cache tiling + register blocking, then threads. OpenBLAS adds " +
      "hand-written assembly micro-kernels - which is why engines call cuBLAS/CUTLASS on GPUs.",
    metric: "gflops", metricLabel: "GFLOP/s",
  },
  gemv: {
    title: "GEMV: y = W·x (decode)",
    why: "Decode at batch 1 multiplies every weight by one vector: 2 FLOPs per weight loaded. The time is set by " +
      "bytes read, so smaller weights (Q8_0 = 8.5 bits, Q4_0 = 4.5 bits) are faster even though they do extra work " +
      "to unpack. This is llama.cpp's whole strategy.",
    metric: "ms", metricLabel: "time (ms) - lower is better",
  },
  attention: {
    title: "Attention: naive vs FlashAttention",
    why: "Naive attention materialises the T×T score matrix. FlashAttention walks K/V in tiles with an online " +
      "softmax, so scratch memory is O(tile) instead of O(T²) and data stays in cache (SRAM on a GPU).",
    metric: "ms", metricLabel: "time (ms) - lower is better",
  },
  softmax: {
    title: "Softmax: 3-pass vs online",
    why: "Online softmax fuses the max and sum passes. On a CPU the win is small (the row fits in cache); on a " +
      "GPU it removes a full HBM pass and is the key trick inside FlashAttention.",
    metric: "ms", metricLabel: "time (ms) - lower is better",
  },
};

export default {
  id: "kernels",
  title: "Kernels",
  render(ctx) {
    const k = ctx.overview.kernels;
    let op = "matmul";
    const n = num(256, 16, 1024);
    const gn = num(4096, 32, 8192);
    const gk = num(4096, 32, 8192);
    const seq = num(512, 16, 2048);
    const heads = num(8, 1, 32);
    const hd = select(["32", "64", "128"], "64");
    const params = h("div", { class: "form" });
    const why = h("div");
    const out = h("div");
    const setOp = (o) => {
      op = o;
      const fields = {
        matmul: [field("Matrix size n (n×n×n)", n)],
        gemv: [field("Output rows N", gn), field("Input dim K (×32)", gk)],
        attention: [field("Sequence length", seq), field("Heads", heads), field("Head dim", hd)],
        softmax: [h("p", { class: "small muted", text: "64 rows × 32,000 columns (one vocabulary of logits per row)" })],
      }[o];
      params.replaceChildren(...fields);
      why.replaceChildren(callout(OPS[o].title, OPS[o].why));
      out.replaceChildren();
    };
    const tabs = seg(Object.keys(OPS).map((o) => [o, o]), op, setOp);
    setOp(op);
    const run = async () => {
      out.replaceChildren(message("info", "Benchmarking (best of several runs)…"));
      const body = { op, n: Number(op === "gemv" ? gn.value : n.value), k: Number(gk.value), seq: Number(seq.value),
        heads: Number(heads.value), head_dim: Number(hd.value) };
      const r = await api("/lab/kernels/bench", body);
      const meta = OPS[op];
      const cols = [{ key: "name", label: "Implementation" }, { key: "ms", label: "Time", num: true, fmt: ms }];
      if (op === "matmul") cols.push({ key: "gflops", label: "GFLOP/s", num: true, fmt: (v) => fmt(v, 1) });
      if (op === "gemv") {
        cols.push({ key: "weight_mb", label: "Weights read", num: true, fmt: (v) => `${fmt(v, 1)} MiB` },
          { key: "gbps", label: "GB/s", num: true, fmt: (v) => fmt(v, 1) },
          { key: "rel_err", label: "rel. error", num: true, fmt: (v) => v.toExponential(1) });
      }
      if (op === "attention") cols.push({ key: "scratch_mb", label: "Scratch memory", num: true, fmt: (v) => `${fmt(v, 2)} MiB` });
      if (op !== "gemv") cols.push({ key: "max_err", label: "max |error|", num: true, fmt: (v) => v.toExponential(1) });
      out.replaceChildren(
        h("p", { class: "small secondary", text: meta.metricLabel }),
        hbars({ categories: r.results.map((x) => x.name), series: [{ name: meta.metricLabel, cls: "c2",
          values: r.results.map((x) => x[meta.metric]) }], valueFmt: (v) => fmt(v, v < 10 ? 2 : 1) }),
        table(cols, r.results));
    };
    return page("Layer 3 · Kernels", "Kernels",
      "The C++ kernels in csrc/ (called through ctypes) implement the same algorithms as the CUDA/Triton kernels " +
      "in vLLM, llama.cpp and FlashAttention - on your CPU, so you can read, change and benchmark them.",
      k.available ? message("ok", `C++ kernels loaded: ${k.version}, SIMD ${k.simd}, OpenMP ${k.openmp ? "on" : "off"}, ` +
        `${k.threads} threads.`) : message("error", `C++ kernels not built (${k.error}). Run "make kernels" - ` +
        "NumPy baselines still run."),
      h("div", { style: { height: "12px" } }),
      card("Benchmark", null, tabs.el, why, params, h("div", { class: "row" }, runButton("Run benchmark", out, run)), out),
      h("div", { style: { height: "16px" } }),
      card("Read the source", "Every kernel is short and commented.",
        table([{ key: 0, label: "File" }, { key: 1, label: "What to look for" }], [
          ["csrc/src/matmul.cpp", "loop order, tiling, 4-row register blocking, AVX2 FMA GEMV"],
          ["csrc/src/attention.cpp", "naive vs FlashAttention online softmax; PagedAttention with bounds-checked block table"],
          ["csrc/src/quant.cpp", "Q8_0/Q4_0 blocks, llama.cpp's maddubs int8 dot-product trick"],
          ["csrc/src/elementwise.cpp", "fused add + RMSNorm, online softmax, RoPE"],
          ["gpu_kernels/cuda/kernels.cu", "shared-memory GEMM, warp-shuffle RMSNorm, FlashAttention, PagedAttention on GPU"],
          ["gpu_kernels/triton/kernels.py", "the same in Triton, with tensor cores via tl.dot"],
        ]),
        code("make kernels          # build with CMake (-O3 -march=native, hardened flags)\n" +
          "make kernels-test     # C++ tests under AddressSanitizer + UBSan\n" +
          "./csrc/build/bench_kernels 1024")));
  },
};
