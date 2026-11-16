"""Pipeline parallelism: split the *layers* into stages on different GPUs.

    GPU0: layers 0-7  ->  GPU1: layers 8-15  ->  GPU2: 16-23  ->  GPU3: 24-31

Only the activation [tokens, d_model] crosses GPUs at stage boundaries
(point-to-point send/recv, cheap enough for inter-node links). But a single
request flows through stages *sequentially*, so to keep every GPU busy the
batch is cut into micro-batches that follow each other through the pipe:

    stage 0 | m0 | m1 | m2 | m3 |    |    |    |
    stage 1 |    | m0 | m1 | m2 | m3 |    |    |
    stage 2 |    |    | m0 | m1 | m2 | m3 |    |
    stage 3 |    |    |    | m0 | m1 | m2 | m3 |
                bubble = (pp - 1) / (m + pp - 1)

PP does not reduce per-token latency (it adds hops); it lets a model that
does not fit on one node run at all, with low communication volume.
"""

from __future__ import annotations

import time

import numpy as np

from inferlab.foundations.model import LlamaModel, LlamaWeights
from inferlab.parallel.cluster import SimCluster


def split_layers(n_layers: int, pp: int) -> list[range]:
    if not 1 <= pp <= n_layers:
        raise ValueError("pp must be between 1 and n_layers")
    base, extra = divmod(n_layers, pp)
    out, start = [], 0
    for s in range(pp):
        n = base + (1 if s < extra else 0)
        out.append(range(start, start + n))
        start += n
    return out


def _stage_model(model: LlamaModel, layers: range) -> LlamaModel:
    from dataclasses import replace

    w = LlamaWeights(model.w.tok_embeddings, [model.w.layers[i] for i in layers], model.w.final_norm,
                     model.w.lm_head, model.w.tied)
    return LlamaModel(replace(model.cfg, n_layers=len(layers)), w, model.kernels)


def _run_stage(stage: LlamaModel, x: np.ndarray, first: bool, last: bool) -> np.ndarray:
    """Run a stage on a hidden-state tensor (or token ids for the first stage)."""
    from inferlab.foundations import ops

    cfg = stage.cfg
    T = x.shape[0]
    cache = stage.new_kv_cache(T)
    ctx = cache.context(T, all_logits=True)
    h = np.asarray(stage.w.tok_embeddings[x], dtype=np.float32) if first else x
    for li, lw in enumerate(stage.w.layers):
        n = ops.rmsnorm(h, lw.attn_norm, cfg.norm_eps)
        q = lw.wq(n).reshape(T, cfg.n_heads, cfg.head_dim)
        k = lw.wk(n).reshape(T, cfg.n_kv_heads, cfg.head_dim)
        v = lw.wv(n).reshape(T, cfg.n_kv_heads, cfg.head_dim)
        q = ops.apply_rope(q, ctx.positions, cfg.rope_theta, cfg.rope_style)
        k = ops.apply_rope(k, ctx.positions, cfg.rope_theta, cfg.rope_style)
        h = h + lw.wo(ctx.attend(li, q, k, v).reshape(T, -1))
        n = ops.rmsnorm(h, lw.ffn_norm, cfg.norm_eps)
        h = h + lw.w2(ops.swiglu(lw.w1(n), lw.w3(n)))
    if last:
        return stage.w.lm_head(ops.rmsnorm(h, stage.w.final_norm, cfg.norm_eps))
    return h


def pp_forward(model: LlamaModel, sequences: list[list[int]], pp: int, num_microbatches: int | None = None):
    """Run prompts through a ``pp``-stage pipeline; returns (logits per seq, schedule, cluster).

    Each micro-batch here is a group of sequences; every sequence is still
    processed on its own (no padding) to keep the code readable.
    """
    if not sequences:
        raise ValueError("need at least one sequence")
    stages_layers = split_layers(model.cfg.n_layers, pp)
    stages = [_stage_model(model, r) for r in stages_layers]
    m = num_microbatches or len(sequences)
    m = max(1, min(m, len(sequences)))
    groups = [list(g) for g in np.array_split(np.arange(len(sequences)), m)]
    cl = SimCluster(pp, link_gbps=50.0)  # e.g. inter-node InfiniBand-class link

    # warm-up pass (first-call allocations would otherwise distort the time-line)
    warm = np.asarray(sequences[0], dtype=np.int64)
    for s, stage in enumerate(stages):
        warm = _run_stage(stage, warm, s == 0, s == pp - 1)

    outputs: dict[int, np.ndarray] = {}
    dur = np.zeros((pp, m))
    for j, g in enumerate(groups):
        acts = {i: np.asarray(sequences[i], dtype=np.int64) for i in g}
        for s, stage in enumerate(stages):
            t0 = time.perf_counter()
            for i in g:
                acts[i] = _run_stage(stage, acts[i], s == 0, s == pp - 1)
            dur[s, j] = time.perf_counter() - t0
            if s < pp - 1:
                for i in g:
                    acts[i] = cl.send_recv(acts[i], tag=f"stage{s}->stage{s + 1}")
        outputs.update(acts)

    # Build the time-line a real pipeline would follow (GPipe forward schedule)
    end = np.zeros((pp, m))
    events = []
    for s in range(pp):
        for j in range(m):
            start = max(end[s - 1, j] if s > 0 else 0.0, end[s, j - 1] if j > 0 else 0.0)
            end[s, j] = start + dur[s, j]
            events.append({"stage": s, "microbatch": j, "start_ms": start * 1e3, "end_ms": end[s, j] * 1e3})
    makespan = float(end[-1, -1])
    busy = float(dur.sum())
    schedule = {
        "stages": [[r.start, r.stop - 1] for r in stages_layers],
        "num_microbatches": m,
        "events": events,
        "makespan_ms": makespan * 1e3,
        "sequential_ms": busy * 1e3,
        "bubble_fraction": 1.0 - busy / (pp * makespan) if makespan > 0 else 0.0,
        "ideal_bubble_fraction": (pp - 1) / (m + pp - 1),
    }
    return [outputs[i] for i in range(len(sequences))], schedule, cl
