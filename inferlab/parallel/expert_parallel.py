"""Expert parallelism for Mixture-of-Experts layers (Mixtral, DeepSeek, Qwen-MoE).

An MoE FFN has E experts; a router sends each token to its top-k. Experts
are spread over ``ep`` GPUs (E/ep each). One MoE layer is:

    route    each rank scores its own tokens: top-k experts + gate weights
    dispatch ALL-TO-ALL: send every token to the ranks owning its experts
    compute  each rank runs its local experts on the tokens it received
    combine  ALL-TO-ALL back, then weighted sum of the k expert outputs

Two all-to-alls per layer, and load balance matters: a "hot" expert makes
its GPU the straggler. ``capacity_factor`` caps tokens per expert - extra
tokens are dropped (output 0 for that expert), trading quality for latency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inferlab.foundations import ops
from inferlab.parallel.cluster import SimCluster

F32 = np.float32


@dataclass
class MoELayer:
    router: np.ndarray  # [E, D]
    w1: np.ndarray  # [E, F, D]
    w3: np.ndarray  # [E, F, D]
    w2: np.ndarray  # [E, D, F]
    top_k: int = 2

    @classmethod
    def random(cls, d: int = 64, f: int = 128, n_experts: int = 8, top_k: int = 2, seed: int = 0,
               skew: float = 0.0) -> MoELayer:
        rng = np.random.default_rng(seed)
        router = rng.normal(0, 1, (n_experts, d)).astype(F32)
        if skew:  # make some experts systematically more popular
            router += skew * rng.normal(0, 1, (n_experts, 1)).astype(F32)
        return cls(
            router,
            rng.normal(0, d**-0.5, (n_experts, f, d)).astype(F32),
            rng.normal(0, d**-0.5, (n_experts, f, d)).astype(F32),
            rng.normal(0, f**-0.5, (n_experts, d, f)).astype(F32),
            top_k,
        )

    @property
    def n_experts(self) -> int:
        return self.router.shape[0]

    def route(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mixtral routing: softmax over the top-k router logits."""
        logits = x @ self.router.T  # [T, E]
        idx = np.argsort(-logits, axis=1, kind="stable")[:, : self.top_k]
        top = np.take_along_axis(logits, idx, axis=1)
        return idx, ops.softmax(top, axis=1).astype(F32)

    def expert(self, e: int, x: np.ndarray) -> np.ndarray:
        return ops.swiglu(x @ self.w1[e].T, x @ self.w3[e].T) @ self.w2[e].T

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Single-device reference."""
        idx, gates = self.route(x)
        out = np.zeros_like(x)
        for e in range(self.n_experts):
            rows, slot = np.nonzero(idx == e)
            if rows.size:
                out[rows] += gates[rows, slot][:, None] * self.expert(e, x[rows])
        return out


def ep_forward(layer: MoELayer, x: np.ndarray, ep: int, capacity_factor: float | None = None,
               cluster: SimCluster | None = None):
    """Expert-parallel forward. Tokens start sharded across ranks (like DP attention)."""
    E = layer.n_experts
    if E % ep:
        raise ValueError(f"n_experts={E} must be divisible by ep={ep}")
    cl = cluster or SimCluster(ep)
    T, D = x.shape
    per_rank = E // ep
    owner = np.arange(E) // per_rank  # which rank holds expert e
    token_shards = np.array_split(np.arange(T), ep)
    capacity = None
    if capacity_factor is not None:
        capacity = max(1, int(np.ceil(capacity_factor * T * layer.top_k / E)))

    # 1) route locally and build per-destination send buffers
    send_x, send_meta = [], []
    for r in range(ep):
        ids = token_shards[r]
        idx, gates = layer.route(x[ids])
        bx, bm = [], []
        for dst in range(ep):
            rows, slot = np.nonzero(owner[idx] == dst)
            bx.append(x[ids][rows])
            bm.append(np.stack([ids[rows], idx[rows, slot], np.arange(len(rows))], 1).astype(np.int64)
                      if len(rows) else np.zeros((0, 3), np.int64))
            bm[-1] = (bm[-1], gates[rows, slot])
        send_x.append(bx)
        send_meta.append(bm)
    recv_x = cl.all_to_all(send_x, tag="moe.dispatch")
    recv_meta = [[send_meta[i][j] for i in range(ep)] for j in range(ep)]  # metadata travels too (tiny)

    # 2) each rank runs its local experts
    tokens_per_expert = np.zeros(E, dtype=np.int64)
    dropped = 0
    back_x = []
    for j in range(ep):
        xs = np.concatenate(recv_x[j], axis=0) if recv_x[j] else np.zeros((0, D), F32)
        metas = np.concatenate([m for m, _ in recv_meta[j]], axis=0)
        gates = np.concatenate([g for _, g in recv_meta[j]], axis=0)
        y = np.zeros_like(xs)
        for e in range(j * per_rank, (j + 1) * per_rank):
            sel = np.nonzero(metas[:, 1] == e)[0]
            tokens_per_expert[e] = len(sel)
            if capacity is not None and len(sel) > capacity:
                dropped += len(sel) - capacity
                sel = sel[:capacity]
            if len(sel):
                y[sel] = gates[sel][:, None] * layer.expert(e, xs[sel])
        # split results back per source rank
        sizes = [len(m) for m, _ in recv_meta[j]]
        back_x.append(list(np.split(y, np.cumsum(sizes)[:-1])) if sizes else [])
    ret = cl.all_to_all(back_x, tag="moe.combine")

    # 3) combine on the source rank
    out = np.zeros_like(x)
    for r in range(ep):
        for dst in range(ep):
            meta, _ = send_meta[r][dst]
            if len(meta):
                np.add.at(out, meta[:, 0], ret[r][dst])
    owner_load = [int(tokens_per_expert[owner == r].sum()) for r in range(ep)]
    stats = {
        "tokens_per_expert": tokens_per_expert.tolist(),
        "tokens_per_rank": owner_load,
        "load_imbalance": float(max(owner_load) / max(np.mean(owner_load), 1e-9)),
        "capacity": capacity,
        "dropped_assignments": int(dropped),
        "comm": cl.stats(),
    }
    return out, stats
