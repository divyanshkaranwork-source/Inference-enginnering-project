"""Lab 10 - 5D parallelism: DP, TP(+SP), PP, CP, EP on a simulated cluster.

Run:  python labs/10_parallelism_5d.py
Read: docs/05-parallelism.md
"""

import numpy as np
from _common import explain, header, table

from inferlab.config import TINY_RANDOM
from inferlab.foundations import ops
from inferlab.foundations.sampling import SamplingParams
from inferlab.foundations.tokenizer import ByteTokenizer
from inferlab.foundations.weights import random_model
from inferlab.parallel import (
    MoELayer,
    Plan,
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

model, tok = random_model(TINY_RANDOM), ByteTokenizer()
ids = tok.encode("Every GPU holds a slice of the work!!")[:32]
ref = model.logits_full(ids)
rng = np.random.default_rng(0)

header("1. Tensor parallelism (+ sequence parallelism)")
for sp in (False, True):
    out, cl = tp_forward(model, ids, 2, sequence_parallel=sp)
    s = cl.stats()
    print(f"  TP=2 SP={sp!s:5}: max err {np.abs(out - ref).max():.1e}, collectives {sorted(s['by_op'])}, "
          f"{s['total_bytes_per_rank'] / 1024:.0f} KiB/rank")
explain("Megatron TP: column-parallel QKV/W1/W3, row-parallel O/W2, one all-reduce after each. SP swaps each "
        "all-reduce for reduce-scatter + all-gather (same bytes) to also split norms and activations.")

header("2. Pipeline parallelism")
outs, sched, _ = pp_forward(model, [tok.encode(f"micro-batch {i}") for i in range(4)], pp=4, num_microbatches=4)
print(f"  4 stages x 4 micro-batches: bubble {sched['bubble_fraction']:.0%} "
      f"(ideal {sched['ideal_bubble_fraction']:.0%}), stages = {sched['stages']}")

header("3. Context parallelism for long sequences")
q, k, v = (rng.standard_normal((256, h, 32), dtype=np.float32) for h in (8, 4, 4))
full = ops.attention(q, k, v, 0, True)
for scheme in ("contiguous", "zigzag"):
    out, st = ring_attention(q, k, v, 4, scheme)
    print(f"  ring {scheme:10}: max err {np.abs(out - full).max():.1e}, load imbalance {st['load_imbalance']:.2f}x")
out, _ = ulysses_attention(q, k, v, 4)
print(f"  ulysses          : max err {np.abs(out - full).max():.1e}")

header("4. Expert parallelism (MoE, 8 experts, top-2)")
layer = MoELayer.random(n_experts=8, top_k=2, skew=1.5)
x = rng.standard_normal((256, 64), dtype=np.float32)
for cap in (None, 1.0):
    out, st = ep_forward(layer, x, 4, cap)
    print(f"  EP=4 capacity={cap}: err {np.abs(out - layer.forward(x)).max():.1e}, tokens/expert "
          f"{st['tokens_per_expert']}, dropped {st['dropped_assignments']}")

header("5. Data parallelism: routing decides cache hits")
for pol in ("round_robin", "prefix_aware"):
    r = simulate_dp(model, tok, make_workload(3, 4), 2, pol, SamplingParams(temperature=0, max_tokens=4))
    print(f"  {pol:13}: assignment {r['assignment']}, prefix-cache hits {r['prefix_hit_tokens']} tokens")

header("6. Planning Llama-2-70B on 8 x H100")
rows = []
for p in (Plan(tp=8), Plan(pp=8), Plan(dp=2, tp=4), Plan(tp=2, pp=4)):
    e = evaluate("llama-2-70b", "H100-SXM-80GB", p)
    rows.append([f"dp{p.dp} tp{p.tp} pp{p.pp}", e["fits"], f"{e['memory_per_gpu_gb']['total']:.1f}",
                 f"{e['decode']['token_latency_ms']:.1f}", f"{e['decode']['tokens_per_s_total']:.0f}"])
table(rows, ["layout", "fits", "GB/GPU", "ms/token", "tok/s total"])
best = recommend("llama-2-70b", "H100-SXM-80GB", 8)[0]["plan"]
print(f"  recommended: dp={best['dp']} tp={best['tp']} pp={best['pp']}")
explain("Decode is memory-bound, so TP (which splits the bytes each GPU reads per token) cuts latency; PP mostly "
        "adds capacity because each micro-batch re-reads its stage's weights. Replicas (DP) scale throughput.")
