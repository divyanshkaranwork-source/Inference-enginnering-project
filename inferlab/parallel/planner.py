"""5D parallelism planner for serving a model on a GPU cluster.

    world = DP x PP x TP x CP          (EP re-uses the DP x TP ranks for MoE layers)

Rules of thumb encoded here (and explained in docs/05-parallelism.md):

* TP shards weights *and* KV heads; it needs 2 all-reduces per layer per step
  -> keep TP inside one NVLink domain (<= 8 GPUs).
* PP shards layers; cheap point-to-point traffic, but adds latency per hop
  and needs micro-batches to avoid bubbles -> use it across nodes.
* CP shards the sequence; only worth it for very long prompts (prefill).
* EP shards MoE experts; all-to-all traffic, load balance matters.
* DP adds replicas; linear throughput scaling, zero model communication.
"""

from __future__ import annotations

from dataclasses import dataclass

from inferlab.config import LlamaConfig, get_model_config
from inferlab.foundations import perfmodel as pm

GB = 1e9


@dataclass
class Plan:
    dp: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1

    @property
    def world(self) -> int:
        return self.dp * self.tp * self.pp * self.cp


def evaluate(
    model: str | LlamaConfig,
    gpu: str,
    plan: Plan,
    weight_dtype: str = "fp16",
    kv_dtype: str = "fp16",
    batch_per_replica: int = 32,
    context_len: int = 4096,
    gpus_per_node: int = 8,
    gpu_util: float = 0.9,
) -> dict:
    cfg = get_model_config(model) if isinstance(model, str) else model
    g = pm.get_gpu(gpu)
    warnings: list[str] = []
    errors: list[str] = []
    for name, v in (("dp", plan.dp), ("tp", plan.tp), ("pp", plan.pp), ("cp", plan.cp), ("ep", plan.ep)):
        if v < 1 or v > 1024:
            errors.append(f"{name} must be in [1, 1024]")
    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}
    if cfg.n_heads % plan.tp:
        errors.append(f"tp={plan.tp} must divide n_heads={cfg.n_heads}")
    if plan.pp > cfg.n_layers:
        errors.append(f"pp={plan.pp} exceeds n_layers={cfg.n_layers}")
    if plan.ep > 1:
        if not cfg.n_experts:
            errors.append("ep > 1 only applies to MoE models (e.g. mixtral-8x7b)")
        elif cfg.n_experts % plan.ep:
            errors.append(f"ep={plan.ep} must divide n_experts={cfg.n_experts}")
        elif plan.ep > plan.dp * plan.tp:
            errors.append("ep must be <= dp * tp (experts are spread over the DP x TP ranks)")
    kv_rep = 1
    if plan.tp > cfg.n_kv_heads:
        kv_rep = plan.tp // cfg.n_kv_heads
        warnings.append(f"tp={plan.tp} > n_kv_heads={cfg.n_kv_heads}: KV heads are replicated {kv_rep}x")
    if plan.tp > gpus_per_node:
        warnings.append("TP spans nodes: all-reduce over the network will dominate latency")
    if plan.cp > 1 and context_len < 32768:
        warnings.append("CP mainly pays off for very long contexts (>= 32k tokens)")

    b = cfg.param_breakdown()
    wb = pm.dtype_bytes(weight_dtype)
    dense = (b["attention"] + b["lm_head"] + b["router"]) * wb + (b["embedding"] + b["norms"]) * 2.0
    expert = b["ffn"] * wb
    expert_shards = plan.tp * (plan.ep if cfg.n_experts else 1)
    w_gpu = dense / (plan.tp * plan.pp) + expert / (expert_shards * plan.pp)

    kv_tok = pm.kv_bytes_per_token(cfg, kv_dtype)
    kv_tok_gpu = kv_tok * kv_rep / (plan.tp * plan.pp * plan.cp)
    tokens = batch_per_replica * context_len
    kv_gpu = kv_tok_gpu * tokens
    act = 4 * min(batch_per_replica * 512, 8192) * max(cfg.dim, cfg.hidden_dim) * 2 / plan.tp
    budget = g.memory_gb * GB * gpu_util
    used = w_gpu + kv_gpu + act
    fits = used <= budget
    if not fits:
        warnings.append("does not fit: raise tp/pp, quantize weights or the KV cache, or reduce batch")
    max_batch = int(max(budget - w_gpu - act, 0) // max(kv_tok_gpu * context_len, 1))

    # ---- decode step (one new token for every sequence of the replica) -----
    # With PP the batch is cut into pp micro-batches that flow through the
    # stages; every stage must re-read its weights once per micro-batch, which
    # is why PP does not speed up memory-bound decode the way TP does.
    bw = g.bandwidth_gbs * 1e9 * 0.8
    peak = g.peak_tflops(weight_dtype) * 1e12 * 0.6
    mb = max(1, batch_per_replica // plan.pp)  # sequences per micro-batch
    w_read = pm.decode_weight_bytes(cfg, weight_dtype, mb) / (plan.tp * plan.pp)
    kv_read = kv_tok_gpu * mb * context_len / 2  # average context = half full
    flops = mb * pm.flops_per_token(cfg, context_len // 2) / (plan.tp * plan.pp)
    t_stage = max((w_read + kv_read) / bw, flops / peak)
    link = (g.interconnect_gbs if plan.tp <= gpus_per_node else 50.0) * 1e9
    act_bytes = mb * cfg.dim * 2
    layers_stage = cfg.n_layers / plan.pp
    t_tp = 0.0
    if plan.tp > 1:  # 2 ring all-reduces per layer
        per_ar = 2 * (plan.tp - 1) / plan.tp * act_bytes / link + 2 * (plan.tp - 1) * 3e-6
        t_tp = 2 * layers_stage * per_ar
    t_ep = 0.0
    if plan.ep > 1 and cfg.n_experts:  # dispatch + combine all-to-all per MoE layer
        a2a = act_bytes * cfg.n_experts_active * (plan.ep - 1) / plan.ep / link + 10e-6
        t_ep = 2 * layers_stage * a2a
    t_pp_hop = act_bytes / (50.0 * 1e9) + 10e-6 if plan.pp > 1 else 0.0
    t_mb = t_stage + t_tp + t_ep
    latency = plan.pp * t_mb + (plan.pp - 1) * t_pp_hop  # one token of one micro-batch
    round_time = plan.pp * t_mb if plan.pp > 1 else latency  # all micro-batches advance once
    throughput_replica = (mb * plan.pp) / round_time

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "model": cfg.name,
        "gpu": g.name,
        "plan": {"dp": plan.dp, "tp": plan.tp, "pp": plan.pp, "cp": plan.cp, "ep": plan.ep, "world": plan.world},
        "memory_per_gpu_gb": {"weights": w_gpu / GB, "kv_cache": kv_gpu / GB, "activations": act / GB,
                              "total": used / GB, "budget": budget / GB},
        "fits": fits,
        "max_batch_per_replica_at_context": max_batch,
        "decode": {
            "token_latency_ms": latency * 1e3,
            "micro_batch": mb,
            "compute_or_memory_ms_per_stage": t_stage * 1e3,
            "tp_comm_ms_per_stage": t_tp * 1e3,
            "ep_comm_ms_per_stage": t_ep * 1e3,
            "pp_hop_ms": t_pp_hop * 1e3,
            "tokens_per_s_per_replica": throughput_replica,
            "tokens_per_s_total": throughput_replica * plan.dp,
            "tokens_per_s_per_user": 1.0 / latency,
        },
    }


def recommend(model: str, gpu: str, n_gpus: int, batch_per_replica: int = 32, context_len: int = 4096,
              weight_dtype: str = "fp16", kv_dtype: str = "fp16", top: int = 5) -> list[dict]:
    """Enumerate DP x TP x PP (x EP) layouts for ``n_gpus`` and rank the ones that fit."""
    if not 1 <= n_gpus <= 512:
        raise ValueError("n_gpus must be in [1, 512]")
    cfg = get_model_config(model)
    out = []
    for tp in (1, 2, 4, 8):
        for pp in (1, 2, 4, 8, 16):
            if n_gpus % (tp * pp):
                continue
            dp = n_gpus // (tp * pp)
            eps = [1] if not cfg.n_experts else [e for e in (1, 2, 4, 8) if e <= dp * tp and cfg.n_experts % e == 0]
            for ep in eps:
                r = evaluate(cfg, gpu, Plan(dp, tp, pp, 1, ep), weight_dtype, kv_dtype, batch_per_replica,
                             context_len)
                if r["valid"] and r["fits"]:
                    out.append(r)
    out.sort(key=lambda r: (-r["decode"]["tokens_per_s_total"], r["decode"]["token_latency_ms"]))
    return out[:top]
