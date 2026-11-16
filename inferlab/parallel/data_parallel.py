"""Data parallelism for serving = replicas + a smart router.

For inference, "DP" means N full copies of the model (each possibly itself
TP/PP-sharded) behind a load balancer. No communication between replicas;
the interesting part is the *routing policy*:

* round_robin   - simple, ignores load and cache contents
* least_loaded  - send to the replica with the fewest outstanding tokens
* prefix_aware  - hash the prompt prefix so requests sharing a system prompt
                  land on the same replica and hit its prefix cache
                  (SGLang router, NVIDIA Dynamo, llm-d use this idea)

``simulate_dp`` runs *real* engines (one per replica, shared weights,
separate KV pools) so the cache-hit numbers are measured, not modelled.
"""

from __future__ import annotations

import hashlib
import time

from inferlab.foundations.model import LlamaModel
from inferlab.foundations.sampling import SamplingParams
from inferlab.foundations.tokenizer import BaseTokenizer
from inferlab.runtime.engine import EngineConfig, LLMEngine

POLICIES = ("round_robin", "least_loaded", "prefix_aware")


class Router:
    def __init__(self, n: int, policy: str, prefix_chars: int = 32):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.n, self.policy, self.prefix_chars = n, policy, prefix_chars
        self._rr = 0

    def pick(self, prompt: str, loads: list[int]) -> int:
        if self.policy == "round_robin":
            r = self._rr % self.n
            self._rr += 1
            return r
        if self.policy == "least_loaded":
            return min(range(self.n), key=lambda i: loads[i])
        h = hashlib.sha256(prompt[: self.prefix_chars].encode("utf-8")).digest()
        return int.from_bytes(h[:8], "little") % self.n


def make_workload(n_tenants: int = 3, per_tenant: int = 4) -> list[str]:
    """Interleaved requests from tenants that each share a long system prompt."""
    systems = [f"System prompt of tenant {t}: you are a careful, friendly assistant #{t}. " * 2
               for t in range(n_tenants)]
    return [systems[t] + f"Question {q}?" for q in range(per_tenant) for t in range(n_tenants)]


def simulate_dp(model: LlamaModel, tok: BaseTokenizer, prompts: list[str], dp: int, policy: str,
                params: SamplingParams, engine_cfg: EngineConfig | None = None, prefix_chars: int = 64) -> dict:
    if not 1 <= dp <= 8:
        raise ValueError("dp must be in [1, 8]")
    engines = [LLMEngine(model, tok, engine_cfg or EngineConfig(block_size=8)) for _ in range(dp)]
    router = Router(dp, policy, prefix_chars)
    assignment = []
    t0 = time.perf_counter()
    # requests arrive one at a time; every arrival the replicas take a step
    for p in prompts:
        # outstanding work per replica = tokens still to compute
        loads = [sum(s.max_total_tokens() - s.num_computed_tokens for s in e.seqs.values()) for e in engines]
        r = router.pick(p, loads)
        assignment.append(r)
        engines[r].add_request(p, params)
        for e in engines:
            if e.has_unfinished():
                e.step()
    steps = [e.step_count for e in engines]
    while any(e.has_unfinished() for e in engines):
        for e in engines:
            if e.has_unfinished():
                e.step()
    wall = time.perf_counter() - t0
    per = []
    for i, e in enumerate(engines):
        s = e.bm.stats()
        per.append({
            "replica": i,
            "requests": assignment.count(i),
            "steps": e.step_count,
            "generated_tokens": e.totals["generation_tokens"],
            "prefix_hit_rate": s["prefix_hit_rate"],
            "prefix_hit_tokens": s["prefix_hits_tokens"],
        })
    total_hits = sum(p["prefix_hit_tokens"] for p in per)
    max_steps = max(p["steps"] for p in per)
    return {
        "dp": dp,
        "policy": policy,
        "assignment": assignment,
        "replicas": per,
        "makespan_steps": max_steps,  # parallel replicas: time ~ the busiest one
        "load_imbalance": max_steps / max(sum(p["steps"] for p in per) / dp, 1e-9),
        "prefix_hit_tokens": total_hits,
        "wall_ms_simulated_serially": wall * 1e3,
        "steps_after_arrivals": steps,
    }
