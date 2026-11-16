"""Observability: Prometheus metrics + a rolling latency summary for the UI.

The serving SLOs everyone tracks:

* TTFT  - time to first token   (prefill + queueing)       -> interactivity
* TPOT  - time per output token  (a.k.a. inter-token latency) -> streaming speed
* E2E   - end-to-end latency  = TTFT + TPOT * (n_tokens - 1)
* throughput (tokens/s), queue depth, KV-cache utilisation, prefix hit rate

Report percentiles (p50/p90/p99), never just averages: tail latency is what
users feel.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily

LAT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
TOK_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1)


class EngineCollector:
    """Exports live engine gauges from the AsyncEngine snapshot at scrape time."""

    def __init__(self, snapshot_fn):
        self.snapshot_fn = snapshot_fn

    def collect(self):
        snap = self.snapshot_fn() or {}
        bs = snap.get("block_stats", {})
        totals = snap.get("totals", {})
        for name, doc, val in (
            ("inferlab_num_running_seqs", "Sequences in the running batch", len(snap.get("running", []))),
            ("inferlab_num_waiting_seqs", "Sequences waiting for admission", snap.get("num_waiting", 0)),
            ("inferlab_kv_cache_usage_ratio", "Fraction of KV blocks in use", bs.get("usage", 0.0)),
            ("inferlab_prefix_cache_hit_ratio", "Prompt tokens served from prefix cache",
             bs.get("prefix_hit_rate", 0.0)),
            ("inferlab_preemptions", "Total preemptions so far", totals.get("preemptions", 0)),
        ):
            g = GaugeMetricFamily(name, doc)
            g.add_metric([], float(val))
            yield g


class Metrics:
    def __init__(self, snapshot_fn=None, window: int = 500):
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("inferlab_requests", "Requests by endpoint and status", ["endpoint", "status"],
                                registry=r)
        self.e2e = Histogram("inferlab_e2e_latency_seconds", "End-to-end request latency", buckets=LAT_BUCKETS,
                             registry=r)
        self.ttft = Histogram("inferlab_ttft_seconds", "Time to first token", buckets=LAT_BUCKETS, registry=r)
        self.tpot = Histogram("inferlab_tpot_seconds", "Time per output token", buckets=TOK_BUCKETS, registry=r)
        self.prompt_tokens = Counter("inferlab_prompt_tokens", "Prompt tokens processed", registry=r)
        self.gen_tokens = Counter("inferlab_generation_tokens", "Tokens generated", registry=r)
        self.inflight = Gauge("inferlab_inflight_requests", "Requests being served", registry=r)
        self.rate_limited = Counter("inferlab_rate_limited", "Requests rejected by the rate limiter", registry=r)
        self.auth_failures = Counter("inferlab_auth_failures", "Requests with a missing or invalid API key", registry=r)
        if snapshot_fn is not None:
            r.register(EngineCollector(snapshot_fn))
        self._lock = threading.Lock()
        self._recent: deque[dict] = deque(maxlen=window)

    def observe_request(self, ttft_s: float | None, e2e_s: float, n_tokens: int, n_prompt: int) -> None:
        self.e2e.observe(e2e_s)
        tpot = None
        if ttft_s is not None:
            self.ttft.observe(ttft_s)
            if n_tokens > 1:
                tpot = (e2e_s - ttft_s) / (n_tokens - 1)
                self.tpot.observe(tpot)
        self.gen_tokens.inc(n_tokens)
        self.prompt_tokens.inc(n_prompt)
        with self._lock:
            self._recent.append({"ttft": ttft_s, "tpot": tpot, "e2e": e2e_s, "tokens": n_tokens})

    def summary(self) -> dict:
        with self._lock:
            rows = list(self._recent)

        def pct(key):
            vals = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
            if vals.size == 0:
                return None
            return {f"p{q}": float(np.percentile(vals, q)) * 1e3 for q in (50, 90, 99)} | {
                "mean": float(vals.mean()) * 1e3}

        return {
            "requests": len(rows),
            "ttft_ms": pct("ttft"),
            "tpot_ms": pct("tpot"),
            "e2e_ms": pct("e2e"),
            "tokens_total": int(sum(r["tokens"] for r in rows)),
        }
