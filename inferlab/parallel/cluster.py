"""A simulated multi-GPU cluster with NCCL-style collectives.

Each "rank" is just a NumPy array in a Python list, so every parallel
algorithm can be checked bit-for-bit against a single-device run. Every
collective records how many bytes each rank sends, using the standard
bandwidth-optimal (ring) cost model:

    all_reduce      2 (N-1)/N * S      (= reduce_scatter + all_gather)
    all_gather        (N-1)/N * S_total
    reduce_scatter    (N-1)/N * S_total
    all_to_all        (N-1)/N * S_local
    send/recv                 S

and estimates time as  bytes / link_bandwidth + latency per step.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class CommEvent:
    op: str
    tag: str
    group_size: int
    bytes_per_rank: float
    est_time_us: float


class SimCluster:
    def __init__(self, world_size: int, link_gbps: float = 450.0, latency_us: float = 5.0):
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        self.world_size = world_size
        self.link_gbps = link_gbps
        self.latency_us = latency_us
        self.events: list[CommEvent] = []

    # ------------------------------------------------------------------
    def _record(self, op: str, tag: str, n: int, bytes_per_rank: float, steps: int) -> None:
        t = bytes_per_rank / (self.link_gbps * 1e9) * 1e6 + self.latency_us * steps
        self.events.append(CommEvent(op, tag, n, float(bytes_per_rank), float(t if n > 1 else 0.0)))

    def all_reduce(self, xs: list[np.ndarray], tag: str = "") -> list[np.ndarray]:
        n = len(xs)
        total = np.sum(np.stack(xs), axis=0)
        self._record("all_reduce", tag, n, 2 * (n - 1) / n * xs[0].nbytes, 2 * (n - 1))
        return [total.copy() for _ in range(n)]

    def all_gather(self, xs: list[np.ndarray], axis: int = 0, tag: str = "") -> list[np.ndarray]:
        n = len(xs)
        full = np.concatenate(xs, axis=axis)
        self._record("all_gather", tag, n, (n - 1) / n * full.nbytes, n - 1)
        return [full.copy() for _ in range(n)]

    def reduce_scatter(self, xs: list[np.ndarray], axis: int = 0, tag: str = "") -> list[np.ndarray]:
        n = len(xs)
        total = np.sum(np.stack(xs), axis=0)
        self._record("reduce_scatter", tag, n, (n - 1) / n * xs[0].nbytes, n - 1)
        return [c.copy() for c in np.array_split(total, n, axis=axis)]

    def all_to_all(self, send: list[list[np.ndarray]], tag: str = "") -> list[list[np.ndarray]]:
        """send[i][j] goes from rank i to rank j; returns recv[j][i]."""
        n = len(send)
        recv = [[send[i][j] for i in range(n)] for j in range(n)]
        per_rank = max(sum(send[i][j].nbytes for j in range(n) if j != i) for i in range(n))
        self._record("all_to_all", tag, n, per_rank, 1)
        return recv

    def send_recv(self, x: np.ndarray, tag: str = "") -> np.ndarray:
        self._record("send_recv", tag, 2, x.nbytes, 1)
        return x.copy()

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        by_op: dict[str, dict] = defaultdict(lambda: {"count": 0, "bytes_per_rank": 0.0, "est_time_us": 0.0})
        for e in self.events:
            d = by_op[e.op]
            d["count"] += 1
            d["bytes_per_rank"] += e.bytes_per_rank
            d["est_time_us"] += e.est_time_us
        return {
            "world_size": self.world_size,
            "link_gbps": self.link_gbps,
            "by_op": dict(by_op),
            "total_bytes_per_rank": sum(e.bytes_per_rank for e in self.events),
            "total_est_time_us": sum(e.est_time_us for e in self.events),
            "num_collectives": len(self.events),
        }

    def reset(self) -> None:
        self.events.clear()
