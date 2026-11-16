"""Tensor parallelism (Megatron-LM) for the Llama model, + sequence parallelism.

Split every weight matrix across ``tp`` GPUs so each holds 1/tp of the model:

    attention:  Wq, Wk, Wv  column-parallel  (each rank owns n_heads/tp heads)
                Wo          row-parallel     -> partial sums -> ALL-REDUCE
    FFN:        W1, W3      column-parallel  (each rank owns hidden_dim/tp)
                W2          row-parallel     -> ALL-REDUCE
    embedding / LM head:    vocab-parallel   (each rank owns vocab/tp rows)

Two all-reduces per layer, on the critical path of *every* token. That is
why TP is used inside one NVLink domain (<= 8 GPUs) and never across slow
links. Sequence parallelism (Megatron-SP) replaces each all-reduce by a
reduce-scatter + all-gather over the *sequence* dimension: the same bytes,
but RMSNorm/residual work and activation memory are split tp ways.
"""

from __future__ import annotations

import numpy as np

from inferlab.foundations import ops
from inferlab.foundations.model import LlamaModel
from inferlab.parallel.cluster import SimCluster

F32 = np.float32


def shard_layer(lw, cfg, tp: int, rank: int) -> dict:
    hd = cfg.head_dim
    hq, hkv = cfg.n_heads // tp, cfg.n_kv_heads // tp
    fq = slice(rank * hq * hd, (rank + 1) * hq * hd)
    fk = slice(rank * hkv * hd, (rank + 1) * hkv * hd)
    ff = slice(rank * cfg.hidden_dim // tp, (rank + 1) * cfg.hidden_dim // tp)
    return {
        "attn_norm": lw.attn_norm,
        "ffn_norm": lw.ffn_norm,
        "wq": lw.wq.dequantize()[fq],  # column-parallel: split output rows
        "wk": lw.wk.dequantize()[fk],
        "wv": lw.wv.dequantize()[fk],
        "wo": lw.wo.dequantize()[:, fq],  # row-parallel: split input columns
        "w1": lw.w1.dequantize()[ff],
        "w3": lw.w3.dequantize()[ff],
        "w2": lw.w2.dequantize()[:, ff],
        "hq": hq,
        "hkv": hkv,
    }


def check_tp(cfg, tp: int) -> None:
    if tp < 1:
        raise ValueError("tp must be >= 1")
    for name, v in (("n_heads", cfg.n_heads), ("n_kv_heads", cfg.n_kv_heads), ("hidden_dim", cfg.hidden_dim)):
        if v % tp:
            raise ValueError(f"{name}={v} is not divisible by tp={tp}")


def tp_forward(model: LlamaModel, tokens, tp: int, sequence_parallel: bool = False,
               cluster: SimCluster | None = None) -> tuple[np.ndarray, SimCluster]:
    """Full-sequence (prefill) forward with tensor parallelism. Returns logits [T, V]."""
    cfg = model.cfg
    check_tp(cfg, tp)
    tokens = np.asarray(tokens, dtype=np.int64)
    T = len(tokens)
    if sequence_parallel and T % tp:
        raise ValueError("sequence parallelism needs T divisible by tp")
    cl = cluster or SimCluster(tp)
    hd = cfg.head_dim
    pos = np.arange(T)
    shards = [[shard_layer(lw, cfg, tp, r) for r in range(tp)] for lw in model.w.layers]

    # vocab-parallel embedding: each rank looks up only ids in its vocab range
    # (Megatron pads the vocabulary so it splits evenly: shard = ceil(V / tp))
    vshard = -(-cfg.vocab_size // tp)
    emb = model.w.tok_embeddings
    parts = []
    for r in range(tp):
        lo, hi = r * vshard, (r + 1) * vshard
        mask = (tokens >= lo) & (tokens < hi)
        e = np.zeros((T, cfg.dim), F32)
        e[mask] = emb[tokens[mask]]
        parts.append(e)
    if sequence_parallel:  # residual stream lives sharded along the sequence
        xs = cl.reduce_scatter(parts, axis=0, tag="embedding")
    else:
        xs = cl.all_reduce(parts, tag="embedding")

    def gather_seq(chunks, tag):
        return cl.all_gather(chunks, axis=0, tag=tag) if sequence_parallel else chunks

    def reduce_out(partials, tag):
        if sequence_parallel:
            return cl.reduce_scatter(partials, axis=0, tag=tag)
        return cl.all_reduce(partials, tag=tag)

    for li, layer in enumerate(shards):
        # --- attention block ---
        hs = [ops.rmsnorm(x, layer[0]["attn_norm"], cfg.norm_eps) for x in xs]  # norm on local shard
        hs = gather_seq(hs, f"L{li}.attn.all_gather")
        partial = []
        for r in range(tp):
            s, h = layer[r], hs[r]
            q = (h @ s["wq"].T).reshape(T, s["hq"], hd)
            k = (h @ s["wk"].T).reshape(T, s["hkv"], hd)
            v = (h @ s["wv"].T).reshape(T, s["hkv"], hd)
            q = ops.apply_rope(q, pos, cfg.rope_theta, cfg.rope_style)
            k = ops.apply_rope(k, pos, cfg.rope_theta, cfg.rope_style)
            a = ops.attention(q, k, v, 0, True).reshape(T, -1)  # local heads only: no comm
            partial.append(a @ s["wo"].T)  # partial sum over this rank's heads
        outs = reduce_out(partial, f"L{li}.attn.reduce")
        xs = [x + o for x, o in zip(xs, outs, strict=True)]
        # --- FFN block ---
        hs = [ops.rmsnorm(x, layer[0]["ffn_norm"], cfg.norm_eps) for x in xs]
        hs = gather_seq(hs, f"L{li}.ffn.all_gather")
        partial = [ops.swiglu(h @ s["w1"].T, h @ s["w3"].T) @ s["w2"].T for s, h in zip(layer, hs, strict=True)]
        outs = reduce_out(partial, f"L{li}.ffn.reduce")
        xs = [x + o for x, o in zip(xs, outs, strict=True)]

    xs = gather_seq(xs, "final.all_gather")
    x = ops.rmsnorm(xs[0], model.w.final_norm, cfg.norm_eps)
    # vocab-parallel LM head: each rank computes logits for its vocab slice
    W = model.w.lm_head.dequantize()
    local = [x @ W[r * vshard : min((r + 1) * vshard, cfg.vocab_size)].T for r in range(tp)]
    logits = cl.all_gather(local, axis=1, tag="lm_head")[0]
    return logits, cl


def tp_memory_per_rank(model: LlamaModel, tp: int) -> dict:
    total = model.nbytes
    return {"full_model_mb": total / 2**20, "per_rank_mb": total / tp / 2**20}
