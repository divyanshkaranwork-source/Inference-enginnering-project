"""/lab/* - interactive experiments behind the web UI.

Every endpoint is authenticated and rate-limited; CPU-heavy ones cost more
rate-limit tokens, run in a worker thread (never on the event loop) and are
serialised by a semaphore so the lab cannot starve the serving engine. All
inputs have hard upper bounds.
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from inferlab.config import ALL_MODELS, REFERENCE_MODELS, TINY_RANDOM
from inferlab.foundations import perfmodel as pm
from inferlab.foundations.generate import generate_text
from inferlab.foundations.model import requantize
from inferlab.foundations.ops import log_softmax, softmax
from inferlab.foundations.quantization import LINEAR_KINDS
from inferlab.foundations.sampling import SamplingParams, distribution_report
from inferlab.foundations.weights import random_model
from inferlab.kernels import bench, cpp
from inferlab.parallel import (
    MoELayer,
    Plan,
    SimCluster,
    ep_forward,
    evaluate,
    pp_forward,
    recommend,
    ring_attention,
    simulate_dp,
    tp_forward,
    ulysses_attention,
)
from inferlab.parallel.data_parallel import make_workload
from inferlab.runtime.backends import ENGINE_PRESETS, NativeBackend
from inferlab.runtime.engine import EngineConfig, LLMEngine
from inferlab.runtime.radix_cache import RadixCache
from inferlab.runtime.speculative import speculative_generate
from inferlab.server.deps import require_auth_and_rate_limit, require_auth_heavy

router = APIRouter(prefix="/lab")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


async def heavy(request: Request, fn, *args):
    """Run CPU-bound lab work in a thread, one job at a time."""
    sem: asyncio.Semaphore = request.app.state.lab_semaphore
    try:
        await asyncio.wait_for(sem.acquire(), timeout=10.0)
    except TimeoutError as exc:
        raise HTTPException(429, "the lab is busy with another experiment - retry shortly") from exc
    try:
        return await run_in_threadpool(fn, *args)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        sem.release()


def _lab(request: Request):
    return request.app.state.lab


def _pieces(tok, ids):
    return [{"id": int(i), "piece": tok.id_to_piece(int(i))} for i in ids]


# ---------------------------------------------------------------------------
@router.get("/overview")
async def overview(request: Request, _p: str = Depends(require_auth_and_rate_limit)):
    lab, s = _lab(request), request.app.state.settings
    cfg = lab.model.cfg
    return {
        "model": cfg.to_dict(),
        "model_bytes": lab.model.nbytes,
        "linear_kind": s.linear_kind,
        "backend": request.app.state.backend.info(),
        "kernels": {"selected": s.kernels, **cpp.info()},
        "reference_models": {k: v.to_dict() for k, v in REFERENCE_MODELS.items()},
        "gpus": {k: vars(v) | {"ridge_point": v.ridge_point} for k, v in pm.GPUS.items()},
        "dtypes": pm.DTYPE_BYTES,
        "security": {
            "auth": request.app.state.auth.enabled,
            "rate_limit_rps": s.rate_limit_rps,
            "rate_limit_burst": s.rate_limit_burst,
            "max_body_bytes": s.max_body_bytes,
            "max_prompt_chars": s.max_prompt_chars,
            "max_tokens": s.max_tokens_limit,
            "request_timeout_s": s.request_timeout_s,
            "cors_origins": s.cors_origins,
            "metrics_public": s.metrics_public,
        },
    }


# ---------------------------------------------------------------------------
# Foundations
# ---------------------------------------------------------------------------
class TokenizeBody(Strict):
    text: str = Field(..., max_length=4000)


@router.post("/tokenize")
async def tokenize(body: TokenizeBody, request: Request, _p: str = Depends(require_auth_and_rate_limit)):
    tok = _lab(request).tokenizer
    ids = tok.encode(body.text, bos=True)
    return {"tokens": _pieces(tok, ids), "n_tokens": len(ids), "n_chars": len(body.text),
            "chars_per_token": len(body.text) / max(len(ids) - 1, 1), "vocab_size": tok.vocab_size}


class SamplingBody(Strict):
    prompt: str = Field(..., min_length=1, max_length=2000)
    temperature: float = Field(1.0, ge=0, le=5)
    top_k: int = Field(0, ge=0, le=1000)
    top_p: float = Field(1.0, gt=0, le=1)
    min_p: float = Field(0.0, ge=0, lt=1)
    repetition_penalty: float = Field(1.0, ge=0.5, le=2)


@router.post("/sampling")
async def sampling(body: SamplingBody, request: Request, _p: str = Depends(require_auth_heavy)):
    lab = _lab(request)

    def run():
        ids = lab.tokenizer.encode(body.prompt)[: lab.model.cfg.max_seq_len - 1]
        logits = lab.model.prefill(np.array(ids), lab.model.new_kv_cache(len(ids)))[-1]
        sp = SamplingParams(temperature=body.temperature, top_k=body.top_k, top_p=body.top_p,
                            min_p=body.min_p, repetition_penalty=body.repetition_penalty).validate()
        rep = distribution_report(logits, sp, ids[1:], top=20)
        rep["pieces"] = [lab.tokenizer.id_to_piece(t) for t in rep["tokens"]]
        return rep

    return await heavy(request, run)


class MemoryBody(Strict):
    model: str = "llama-2-7b"
    weight_dtype: str = "fp16"
    kv_dtype: str = "fp16"
    batch: int = Field(8, ge=1, le=4096)
    seq_len: int = Field(4096, ge=1, le=1_048_576)
    gpu: str = "A100-SXM-80GB"
    n_gpus: int = Field(1, ge=1, le=1024)


def _cfg(name: str):
    if name not in ALL_MODELS:
        raise HTTPException(400, f"unknown model; choose from {sorted(ALL_MODELS)}")
    return ALL_MODELS[name]


@router.post("/memory")
async def memory(body: MemoryBody, request: Request, _p: str = Depends(require_auth_and_rate_limit)):
    try:
        return pm.memory_report(_cfg(body.model), body.weight_dtype, body.kv_dtype, body.batch, body.seq_len,
                                body.gpu, n_gpus=body.n_gpus)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class RooflineBody(Strict):
    model: str = "llama-2-7b"
    gpu: str = "A100-SXM-80GB"
    weight_dtype: str = "fp16"


@router.post("/roofline")
async def roofline(body: RooflineBody, request: Request, _p: str = Depends(require_auth_and_rate_limit)):
    try:
        return pm.roofline_chart(_cfg(body.model), body.gpu, body.weight_dtype)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class KVBody(Strict):
    prompt: str = Field("Once upon a time, there was a little robot who", min_length=1, max_length=1000)
    new_tokens: int = Field(48, ge=4, le=96)


@router.post("/kv-cache")
async def kv_cache_experiment(body: KVBody, request: Request, _p: str = Depends(require_auth_heavy)):
    lab = _lab(request)

    def run():
        sp = SamplingParams(temperature=0.0, max_tokens=body.new_tokens, ignore_eos=True)
        t_cache, s_cache = generate_text(lab.model, lab.tokenizer, body.prompt, sp, use_kv_cache=True)
        t_none, s_none = generate_text(lab.model, lab.tokenizer, body.prompt, sp, use_kv_cache=False)
        cfg = lab.model.cfg
        return {
            "with_cache": s_cache.to_dict(),
            "without_cache": s_none.to_dict(),
            "same_output": t_cache == t_none,
            "text": t_cache,
            "kv_bytes_per_token": 2 * cfg.n_layers * cfg.kv_dim * 4,
            "llama2_7b_kv_bytes_per_token_fp16": pm.kv_bytes_per_token(REFERENCE_MODELS["llama-2-7b"]),
        }

    return await heavy(request, run)


class QuantBody(Strict):
    kinds: list[str] = Field(default_factory=lambda: ["fp32", "fp16", "int8", "q8_0", "q4_0"], max_length=6)
    prompt: str = Field("Once upon a time, there was a little girl named Lily. She loved", max_length=500)
    new_tokens: int = Field(24, ge=1, le=64)


@router.post("/quantization")
async def quantization(body: QuantBody, request: Request, _p: str = Depends(require_auth_heavy)):
    lab = _lab(request)
    if any(k not in LINEAR_KINDS for k in body.kinds):
        raise HTTPException(400, f"kinds must be from {LINEAR_KINDS}")
    if lab.model.cfg.n_params > 150_000_000:
        raise HTTPException(400, "quantization lab is limited to models <= 150M parameters")

    def run():
        ids = np.array(lab.tokenizer.encode(body.prompt)[: lab.model.cfg.max_seq_len // 2])
        base = lab.model.logits_full(ids)
        p_ref = softmax(base.astype(np.float64), axis=-1)
        rows = []
        sp = SamplingParams(temperature=0.0, max_tokens=body.new_tokens)
        for kind in body.kinds:
            q = lab.model if kind == "fp32" and lab.settings.linear_kind == "fp32" else requantize(lab.model, kind)
            t0 = time.perf_counter()
            lg = q.logits_full(ids)
            dt = time.perf_counter() - t0
            lp = log_softmax(lg.astype(np.float64), axis=-1)
            kl = float(np.mean(np.sum(p_ref * (np.log(np.clip(p_ref, 1e-30, 1)) - lp), axis=-1)))
            text, _ = generate_text(q, lab.tokenizer, body.prompt, sp)
            rows.append({
                "kind": kind,
                "weights_mb": q.nbytes / 2**20,
                "bits_per_weight": {"fp32": 32, "fp16": 16, "bf16": 16, "int8": 8, "q8_0": 8.5, "q4_0": 4.5}[kind],
                "kl_divergence": kl,
                "top1_agreement": float(np.mean(np.argmax(lg, -1) == np.argmax(base, -1))),
                "max_logit_error": float(np.max(np.abs(lg - base))),
                "prefill_ms": dt * 1e3,
                "sample": text,
            })
        return {"prompt_tokens": len(ids), "results": rows}

    return await heavy(request, run)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
@router.get("/engine/state")
async def engine_state(request: Request, _p: str = Depends(require_auth_and_rate_limit)):
    b = request.app.state.backend
    if not isinstance(b, NativeBackend):
        raise HTTPException(400, "live engine state is only available with the native backend")
    return b.async_engine.snapshot()


class BurstBody(Strict):
    n: int = Field(8, ge=1, le=16)
    max_tokens: int = Field(96, ge=1, le=128)
    shared_prefix: bool = True
    stagger_ms: int = Field(40, ge=0, le=1000)
    temperature: float = Field(0.8, ge=0, le=2)


BURST_PROMPTS = [
    "Once upon a time, there was a little dog named Max.",
    "Lily and her mom went to the park on a sunny day.",
    "Tom had a red ball. He liked to play with it every day.",
    "The little bird wanted to fly high in the sky.",
    "One day, a girl found a shiny stone in the garden.",
    "Sam was sad because his toy car was broken.",
]


@router.post("/engine/burst")
async def engine_burst(body: BurstBody, request: Request, _p: str = Depends(require_auth_heavy)):
    """Fire N concurrent requests into the live engine to watch continuous batching."""
    b = request.app.state.backend
    if not isinstance(b, NativeBackend):
        raise HTTPException(400, "burst requires the native backend")
    tasks: set = request.app.state.burst_tasks
    if len(tasks) >= 32:
        raise HTTPException(429, "too many burst requests in flight")
    prefix = "Once upon a time, in a small village near a big forest, there lived many happy animals. " \
        if body.shared_prefix else ""
    ids = []

    async def consume(prompt, rid, delay):
        # requests arrive staggered, like real traffic: early ones finish their
        # prefill before later ones arrive, so later ones hit the prefix cache
        await asyncio.sleep(delay)
        try:
            async for _ in b.async_engine.generate(prompt, SamplingParams(temperature=body.temperature,
                                                                          max_tokens=body.max_tokens), rid):
                pass
        except Exception:  # noqa: BLE001 - background demo traffic; errors are visible in engine state
            return

    for i in range(body.n):
        rid = f"burst-{int(time.time() * 1000) % 10**8}-{i}"
        prompt = prefix + BURST_PROMPTS[i % len(BURST_PROMPTS)]
        t = asyncio.create_task(consume(prompt, rid, i * body.stagger_ms / 1000))
        tasks.add(t)
        t.add_done_callback(tasks.discard)
        ids.append(rid)
    return {"submitted": ids}


class BatchingBody(Strict):
    num_requests: int = Field(8, ge=2, le=16)
    max_tokens: int = Field(32, ge=4, le=64)
    max_num_seqs: int = Field(4, ge=1, le=16)


@router.post("/batching-compare")
async def batching_compare(body: BatchingBody, request: Request, _p: str = Depends(require_auth_heavy)):
    """Same workload under static vs continuous batching (offline engines)."""
    lab = _lab(request)

    def run():
        rng = np.random.default_rng(0)
        lens = rng.integers(4, body.max_tokens + 1, body.num_requests)  # mixed output lengths
        prompts = [BURST_PROMPTS[i % len(BURST_PROMPTS)] for i in range(body.num_requests)]
        out = {}
        for policy in ("static", "continuous"):
            eng = LLMEngine(lab.model, lab.tokenizer, EngineConfig(num_kv_blocks=64, block_size=16,
                                                                   max_num_seqs=body.max_num_seqs,
                                                                   policy=policy))
            t0 = time.perf_counter()
            seqs = [eng.add_request(p, SamplingParams(temperature=0, max_tokens=int(n), ignore_eos=True))
                    for p, n in zip(prompts, lens, strict=True)]
            batch_sizes = []
            while eng.has_unfinished():
                eng.step()
                batch_sizes.append(eng.history[-1].num_seqs if eng.history else 0)
            wall = time.perf_counter() - t0
            m = [s.metrics.to_dict() for s in seqs]
            out[policy] = {
                "wall_ms": wall * 1e3,
                "steps": eng.step_count,
                "tokens": int(lens.sum()),
                "tokens_per_s": float(lens.sum()) / wall,
                "mean_ttft_ms": float(np.mean([x["ttft_ms"] for x in m])),
                "mean_e2e_ms": float(np.mean([x["e2e_ms"] for x in m])),
                "batch_size_per_step": batch_sizes,
                "per_request": [{"max_tokens": int(n), **x} for n, x in zip(lens, m, strict=True)],
            }
        return out

    return await heavy(request, run)


class RadixBody(Strict):
    prompts: list[str] = Field(..., min_length=1, max_length=16)
    capacity_tokens: int = Field(4096, ge=16, le=65536)


@router.post("/radix")
async def radix(body: RadixBody, request: Request, _p: str = Depends(require_auth_and_rate_limit)):
    if any(len(p) > 2000 for p in body.prompts):
        raise HTTPException(400, "each prompt must be <= 2000 characters")
    tok = _lab(request).tokenizer
    rc = RadixCache(body.capacity_tokens)
    matched = []
    for p in body.prompts:
        ids = tok.encode(p, bos=True)
        matched.append({"tokens": len(ids), "cached": rc.insert(ids)})
    return {"inserts": matched, **rc.to_dict(decode=lambda k: tok.decode(k))}


class SpecBody(Strict):
    prompt: str = Field("Once upon a time, there was a little girl named Lily. Lily liked to play.",
                        min_length=1, max_length=2000)
    method: Literal["ngram", "draft_model"] = "ngram"
    k: int = Field(4, ge=1, le=8)
    max_tokens: int = Field(64, ge=1, le=128)
    draft_layers: int = Field(2, ge=1, le=64)
    temperature: float = Field(0.0, ge=0, le=2)


@router.post("/speculative")
async def speculative(body: SpecBody, request: Request, _p: str = Depends(require_auth_heavy)):
    lab = _lab(request)

    def run():
        sp = SamplingParams(temperature=body.temperature, max_tokens=body.max_tokens, seed=0)
        base_text, base = generate_text(lab.model, lab.tokenizer, body.prompt, sp)
        draft = None
        if body.method == "draft_model":
            draft = lab.model.truncated(min(body.draft_layers, lab.model.cfg.n_layers))
        out, st = speculative_generate(lab.model, lab.tokenizer, body.prompt, sp, body.method, draft, body.k)
        d = st.to_dict()
        for r in d["trace"]:
            r["proposed_pieces"] = [lab.tokenizer.id_to_piece(t) for t in r["proposed"]]
            r["emitted_pieces"] = [lab.tokenizer.id_to_piece(t) for t in r["tokens"]]
        return {
            "baseline": {"text": base_text, **base.to_dict()},
            "speculative": {"text": lab.tokenizer.decode(out), **d},
            "identical_output": lab.tokenizer.decode(out) == base_text if body.temperature == 0 else None,
            "speedup": (base.total_s * 1e3) / d["total_ms"] if d["total_ms"] else None,
        }

    return await heavy(request, run)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
class BenchBody(Strict):
    op: Literal["matmul", "gemv", "attention", "softmax"]
    n: int = Field(256, ge=16, le=8192)
    k: int = Field(4096, ge=32, le=8192)
    seq: int = Field(512, ge=16, le=2048)
    heads: int = Field(8, ge=1, le=32)
    head_dim: Literal[32, 64, 128] = 64


@router.post("/kernels/bench")
async def kernels_bench(body: BenchBody, request: Request, _p: str = Depends(require_auth_heavy)):
    def run():
        if body.op == "matmul":
            return bench.bench_matmul(min(body.n, bench.MAX_GEMM))
        if body.op == "gemv":
            return bench.bench_gemv(body.n, body.k - body.k % 32)
        if body.op == "attention":
            return bench.bench_attention(body.seq, body.heads, body.head_dim)
        return bench.bench_softmax(64, 32000)

    return await heavy(request, run)


# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------
class PlanBody(Strict):
    model: str = "llama-2-7b"
    gpu: str = "A100-SXM-80GB"
    dp: int = Field(1, ge=1, le=1024)
    tp: int = Field(1, ge=1, le=64)
    pp: int = Field(1, ge=1, le=128)
    cp: int = Field(1, ge=1, le=64)
    ep: int = Field(1, ge=1, le=64)
    weight_dtype: str = "fp16"
    kv_dtype: str = "fp16"
    batch: int = Field(32, ge=1, le=4096)
    context_len: int = Field(4096, ge=128, le=1_048_576)


@router.post("/parallel/plan")
async def parallel_plan(body: PlanBody, request: Request, _p: str = Depends(require_auth_and_rate_limit)):
    try:
        return evaluate(_cfg(body.model), body.gpu, Plan(body.dp, body.tp, body.pp, body.cp, body.ep),
                        body.weight_dtype, body.kv_dtype, body.batch, body.context_len)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class RecommendBody(Strict):
    model: str = "llama-2-70b"
    gpu: str = "H100-SXM-80GB"
    n_gpus: int = Field(8, ge=1, le=512)
    batch: int = Field(32, ge=1, le=4096)
    context_len: int = Field(4096, ge=128, le=1_048_576)
    weight_dtype: str = "fp16"


@router.post("/parallel/recommend")
async def parallel_recommend(body: RecommendBody, request: Request, _p: str = Depends(require_auth_heavy)):
    _cfg(body.model)
    return await heavy(request, recommend, body.model, body.gpu, body.n_gpus, body.batch, body.context_len,
                       body.weight_dtype)


class SimBody(Strict):
    strategy: Literal["tp", "sp", "pp", "cp_ring", "cp_ulysses", "ep", "dp"]
    degree: int = Field(2, ge=1, le=8)
    microbatches: int = Field(4, ge=1, le=16)
    zigzag: bool = False
    capacity_factor: float | None = Field(None, gt=0, le=8)
    skew: float = Field(0.0, ge=0, le=5)
    dp_policy: Literal["round_robin", "least_loaded", "prefix_aware"] = "prefix_aware"


@router.post("/parallel/simulate")
async def parallel_simulate(body: SimBody, request: Request, _p: str = Depends(require_auth_heavy)):
    lab = _lab(request)

    def run():
        m = lab.sim_model
        tok = lab.sim_tokenizer
        d = body.degree
        if body.strategy in ("tp", "sp"):
            ids = tok.encode("Tensor parallelism splits every matrix!")[: 32 - 32 % max(d, 1)] or [1]
            ids = ids[: len(ids) - len(ids) % d] if body.strategy == "sp" else ids
            ref = m.logits_full(ids)
            out, cl = tp_forward(m, ids, d, sequence_parallel=body.strategy == "sp")
            return {"max_abs_error": float(np.abs(out - ref).max()), "tokens": len(ids), "comm": cl.stats(),
                    "per_rank_weight_mb": m.nbytes / d / 2**20, "full_weight_mb": m.nbytes / 2**20}
        if body.strategy == "pp":
            seqs = [tok.encode(p) for p in BURST_PROMPTS[:max(body.microbatches, 1)]]
            outs, sched, cl = pp_forward(m, seqs, d, body.microbatches)
            err = max(float(np.abs(o - m.logits_full(s)).max()) for o, s in zip(outs, seqs, strict=True))
            return {"max_abs_error": err, "schedule": sched, "comm": cl.stats()}
        rng = np.random.default_rng(0)
        if body.strategy in ("cp_ring", "cp_ulysses"):
            T, H, KV, hd = 128, 8, 4, 32
            q = rng.standard_normal((T, H, hd), dtype=np.float32)
            k = rng.standard_normal((T, KV, hd), dtype=np.float32)
            v = rng.standard_normal((T, KV, hd), dtype=np.float32)
            from inferlab.foundations.ops import attention

            ref = attention(q, k, v, 0, True)
            if body.strategy == "cp_ring":
                out, st = ring_attention(q, k, v, d, "zigzag" if body.zigzag else "contiguous", SimCluster(d))
            else:
                out, st = ulysses_attention(q, k, v, d, SimCluster(d))
            return {"max_abs_error": float(np.abs(out - ref).max()), "seq_len": T, **st}
        if body.strategy == "ep":
            layer = MoELayer.random(n_experts=8, top_k=2, skew=body.skew)
            x = rng.standard_normal((128, 64), dtype=np.float32)
            out, st = ep_forward(layer, x, d, body.capacity_factor)
            return {"max_abs_error": float(np.abs(out - layer.forward(x)).max()), "tokens": 128, **st}
        prompts = make_workload(3, 4)
        return simulate_dp(m, tok, prompts, d, body.dp_policy, SamplingParams(temperature=0, max_tokens=8))

    return await heavy(request, run)


# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------
@router.get("/engines")
async def engines(request: Request, _p: str = Depends(require_auth_and_rate_limit)):
    b = request.app.state.backend
    health = await b.health()
    return {"active": b.info() | {"health": health}, "presets": ENGINE_PRESETS}


@router.get("/metrics/summary")
async def metrics_summary(request: Request, _p: str = Depends(require_auth_and_rate_limit)):
    return request.app.state.metrics.summary()


class LabContext:
    """Models used by the lab (the live model + a tiny one for parallel sims)."""

    def __init__(self, model, tokenizer, settings):
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings
        self.sim_model = random_model(TINY_RANDOM, seed=0)
        from inferlab.foundations.tokenizer import ByteTokenizer

        self.sim_tokenizer = ByteTokenizer()
