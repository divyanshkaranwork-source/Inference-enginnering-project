"""Back-of-the-envelope performance model: memory, FLOPs, roofline.

The single most useful skill in inference engineering is estimating, before
writing any code, whether a workload is **compute-bound** or
**memory-bandwidth-bound**. This module encodes the standard formulas.

Key facts for a dense decoder (N parameters, L layers, d model width):

* FLOPs per generated token  ~= 2 N  (+ 4 L d * context for attention)
* Bytes read per decode step  =  weights  +  KV cache of every sequence
* Arithmetic intensity (AI)   =  FLOPs / bytes.
  Decode at batch 1 with fp16 weights has AI ~= 1 FLOP/byte, while an
  H100 needs ~300 FLOP/byte to be compute-bound -> decode is memory-bound
  and batching is (almost) free throughput until AI reaches the ridge point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from inferlab.config import LlamaConfig

GB = 1e9
GiB = 1024**3

# bytes per element. q4_0 = 18 bytes / 32 weights, q8_0 = 34 / 32, q4_k_m ~ 4.85 bits
DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "q8_0": 34 / 32,
    "int4": 0.5,
    "q4_0": 18 / 32,
    "q4_k_m": 4.85 / 8,
}


@dataclass(frozen=True)
class GPUSpec:
    name: str
    memory_gb: float
    bandwidth_gbs: float  # HBM / DRAM bandwidth (GB/s)
    tflops_fp16: float  # dense tensor-core FP16/BF16 TFLOPS
    tflops_fp8: float  # dense FP8 (or INT8) TFLOPS, 0 if unsupported
    interconnect_gbs: float  # per-GPU NVLink (or PCIe) bandwidth, one direction
    notes: str = ""

    def peak_tflops(self, dtype: str) -> float:
        if dtype in ("fp8", "int8") and self.tflops_fp8 > 0:
            return self.tflops_fp8
        return self.tflops_fp16

    @property
    def ridge_point(self) -> float:
        """FLOP/byte at which the GPU turns from memory- to compute-bound (fp16)."""
        return self.tflops_fp16 * 1e12 / (self.bandwidth_gbs * 1e9)


# Approximate public datasheet numbers (dense, no sparsity). Good to ~10-20%.
GPUS: dict[str, GPUSpec] = {
    g.name: g
    for g in (
        GPUSpec("H100-SXM-80GB", 80, 3350, 989, 1979, 450, "NVLink4 900 GB/s bidirectional"),
        GPUSpec("H200-SXM-141GB", 141, 4800, 989, 1979, 450, "HBM3e"),
        GPUSpec("B200-180GB", 180, 8000, 2250, 4500, 900, "NVLink5"),
        GPUSpec("A100-SXM-80GB", 80, 2039, 312, 624, 300, "INT8 624 TOPS, no FP8"),
        GPUSpec("L40S-48GB", 48, 864, 362, 733, 32, "PCIe Gen4"),
        GPUSpec("L4-24GB", 24, 300, 121, 242, 32, "PCIe Gen4"),
        GPUSpec("A10G-24GB", 24, 600, 125, 250, 32, "AWS g5"),
        GPUSpec("RTX-4090-24GB", 24, 1008, 165, 330, 32, "consumer, PCIe"),
        GPUSpec("Apple-M2-Max", 64, 400, 27, 0, 0, "unified memory (llama.cpp Metal)"),
        GPUSpec("CPU-DDR5-8ch", 512, 300, 5, 0, 0, "server CPU, AVX-512"),
        GPUSpec("Laptop-CPU", 16, 50, 0.5, 0, 0, "the machine this demo runs on (rough)"),
    )
}


def get_gpu(name: str) -> GPUSpec:
    try:
        return GPUS[name]
    except KeyError:
        raise ValueError(f"unknown GPU {name!r}; choose from {sorted(GPUS)}") from None


def dtype_bytes(dtype: str) -> float:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError:
        raise ValueError(f"unknown dtype {dtype!r}; choose from {sorted(DTYPE_BYTES)}") from None


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
def weight_bytes(cfg: LlamaConfig, dtype: str = "fp16") -> float:
    """Bytes to *store* the weights. Norms/embeddings stay 16-bit when quantizing."""
    b = cfg.param_breakdown()
    hi_bytes = 4.0 if dtype == "fp32" else 2.0
    hi = (b["norms"] + b["embedding"]) * hi_bytes
    lo = (b["attention"] + b["ffn"] + b["router"] + b["lm_head"]) * dtype_bytes(dtype)
    return hi + lo


def decode_weight_bytes(cfg: LlamaConfig, dtype: str = "fp16", batch: int = 1) -> float:
    """Bytes of weights *read* by one decode step.

    The embedding table is only gathered (a few rows), but the LM head is read
    in full. For MoE models only the experts picked by at least one token in
    the batch are read: E * (1 - (1 - k/E)^batch) experts in expectation.
    """
    b = cfg.param_breakdown()
    hi_bytes = 4.0 if dtype == "fp32" else 2.0
    head = b["lm_head"] if not cfg.tie_embeddings else b["embedding"]
    ffn = b["ffn"]
    if cfg.n_experts:
        e, k = cfg.n_experts, cfg.n_experts_active
        touched = e * (1 - (1 - k / e) ** max(batch, 1))
        ffn = ffn / e * touched
    return b["norms"] * hi_bytes + (b["attention"] + ffn + b["router"] + head) * dtype_bytes(dtype)


def kv_bytes_per_token(cfg: LlamaConfig, kv_dtype: str = "fp16") -> float:
    """2 (K and V) x layers x kv_heads x head_dim x bytes."""
    return 2 * cfg.n_layers * cfg.kv_dim * dtype_bytes(kv_dtype)


def kv_cache_bytes(cfg: LlamaConfig, batch: int, seq_len: int, kv_dtype: str = "fp16") -> float:
    return batch * seq_len * kv_bytes_per_token(cfg, kv_dtype)


def memory_report(
    cfg: LlamaConfig,
    weight_dtype: str = "fp16",
    kv_dtype: str = "fp16",
    batch: int = 1,
    seq_len: int = 4096,
    gpu: str = "A100-SXM-80GB",
    gpu_util: float = 0.9,
    n_gpus: int = 1,
) -> dict:
    """How much memory does serving need, and how many sequences fit?"""
    g = get_gpu(gpu)
    w = weight_bytes(cfg, weight_dtype)
    kv_tok = kv_bytes_per_token(cfg, kv_dtype)
    kv = kv_tok * batch * seq_len
    # activation workspace: a few [tokens, max(d, f)] buffers for the largest chunk
    act = 4 * min(batch * seq_len, 8192) * max(cfg.dim, cfg.hidden_dim) * 2
    budget = g.memory_gb * GB * gpu_util * n_gpus
    free_for_kv = max(budget - w - act, 0.0)
    max_tokens = int(free_for_kv // kv_tok) if kv_tok else 0
    return {
        "model": cfg.name,
        "gpu": g.name,
        "n_gpus": n_gpus,
        "params_billion": cfg.n_params / 1e9,
        "weights_gb": w / GB,
        "kv_cache_gb": kv / GB,
        "kv_per_token_kb": kv_tok / 1024,
        "kv_per_sequence_gb": kv_tok * seq_len / GB,
        "activations_gb": act / GB,
        "total_gb": (w + kv + act) / GB,
        "budget_gb": budget / GB,
        "fits": w + kv + act <= budget,
        "max_kv_tokens": max_tokens,
        "max_concurrent_seqs_at_seq_len": max_tokens // max(seq_len, 1),
    }


# ---------------------------------------------------------------------------
# FLOPs, arithmetic intensity, roofline
# ---------------------------------------------------------------------------
def flops_per_token(cfg: LlamaConfig, context_len: int = 0) -> float:
    """2 FLOPs (multiply + add) per active weight + attention over the context."""
    linear = 2 * (cfg.n_active_params - cfg.param_breakdown()["embedding"])
    attn = 4 * cfg.n_layers * cfg.dim * context_len  # QK^T and PV
    return float(linear + attn)


def roofline_point(flops: float, bytes_moved: float, gpu: GPUSpec, dtype: str = "fp16") -> dict:
    ai = flops / max(bytes_moved, 1.0)
    peak = gpu.peak_tflops(dtype) * 1e12
    bw = gpu.bandwidth_gbs * 1e9
    attainable = min(peak, ai * bw)
    t_compute = flops / peak
    t_memory = bytes_moved / bw
    return {
        "flops": flops,
        "bytes": bytes_moved,
        "arithmetic_intensity": ai,
        "attainable_tflops": attainable / 1e12,
        "time_s": max(t_compute, t_memory),
        "time_compute_s": t_compute,
        "time_memory_s": t_memory,
        "bound": "compute" if t_compute >= t_memory else "memory",
    }


def decode_step(
    cfg: LlamaConfig,
    gpu: str,
    batch: int = 1,
    context_len: int = 1024,
    weight_dtype: str = "fp16",
    kv_dtype: str = "fp16",
    n_gpus: int = 1,
    efficiency: float = 0.7,
) -> dict:
    """One decode iteration for ``batch`` sequences (ideal TP over ``n_gpus``)."""
    g = get_gpu(gpu)
    flops = batch * flops_per_token(cfg, context_len)
    bytes_moved = decode_weight_bytes(cfg, weight_dtype, batch) + kv_cache_bytes(
        cfg, batch, context_len, kv_dtype
    )
    p = roofline_point(flops / n_gpus, bytes_moved / n_gpus, g, weight_dtype)
    t = p["time_s"] / efficiency
    p.update(
        step_time_ms=t * 1e3,
        tokens_per_s=batch / t,
        tokens_per_s_per_seq=1.0 / t,
        batch=batch,
        phase="decode",
    )
    return p


def prefill(
    cfg: LlamaConfig,
    gpu: str,
    prompt_len: int = 1024,
    batch: int = 1,
    weight_dtype: str = "fp16",
    n_gpus: int = 1,
    efficiency: float = 0.6,
) -> dict:
    """Process ``prompt_len`` tokens at once: weights are reused across tokens."""
    g = get_gpu(gpu)
    tokens = prompt_len * batch
    flops = tokens * flops_per_token(cfg, prompt_len // 2)  # average causal context
    bytes_moved = decode_weight_bytes(cfg, weight_dtype, tokens) + kv_cache_bytes(cfg, batch, prompt_len)
    p = roofline_point(flops / n_gpus, bytes_moved / n_gpus, g, weight_dtype)
    t = p["time_s"] / efficiency
    p.update(ttft_ms=t * 1e3, tokens_per_s=tokens / t, phase="prefill", batch=batch)
    return p


def roofline_chart(cfg: LlamaConfig, gpu: str, weight_dtype: str = "fp16") -> dict:
    """Points for the UI roofline plot: decode at several batch sizes + prefill."""
    g = get_gpu(gpu)
    pts = []
    for b in (1, 8, 32, 128, 512):
        p = decode_step(cfg, gpu, batch=b, context_len=1024, weight_dtype=weight_dtype)
        pts.append({"label": f"decode bs={b}", **p})
    for n in (128, 2048):
        p = prefill(cfg, gpu, prompt_len=n, weight_dtype=weight_dtype)
        pts.append({"label": f"prefill {n} tok", **p})
    return {
        "gpu": asdict(g),
        "ridge_point": g.ridge_point,
        "peak_tflops": g.peak_tflops(weight_dtype),
        "bandwidth_gbs": g.bandwidth_gbs,
        "points": pts,
    }
