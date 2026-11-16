"""Lab 06 - PagedAttention + continuous batching (the vLLM engine design).

Run:  python labs/06_continuous_batching.py
Read: docs/02-runtime.md (sections 1-4)
"""

import time

import numpy as np
from _common import explain, header, load, table

from inferlab.foundations.sampling import SamplingParams
from inferlab.runtime.engine import EngineConfig, LLMEngine

model, tok = load()
prompts = ["Tom had a red ball.", "The sun was hot.", "Lily saw a big dog in the park.", "Max liked to run.",
           "One day a bird flew into the garden.", "Sam was sad.", "The cat jumped on the bed.", "Mia had a cake."]
lens = np.random.default_rng(0).integers(8, 64, len(prompts))  # very different output lengths


def run(policy, max_num_seqs=4, **kw):
    eng = LLMEngine(model, tok, EngineConfig(policy=policy, max_num_seqs=max_num_seqs, **kw))
    seqs = [eng.add_request(p, SamplingParams(temperature=0, max_tokens=int(n), ignore_eos=True))
            for p, n in zip(prompts, lens, strict=True)]
    t0 = time.perf_counter()
    while eng.has_unfinished():
        eng.step()
    wall = time.perf_counter() - t0
    m = [s.metrics.to_dict() for s in seqs]
    return eng, wall, m


header("1. Static vs continuous batching (same 8 requests, batch of 4)")
rows = []
for policy in ("static", "continuous"):
    eng, wall, m = run(policy)
    rows.append([policy, eng.step_count, f"{lens.sum() / wall:.0f}", f"{np.mean([x['ttft_ms'] for x in m]):.1f}",
                 f"{np.mean([x['e2e_ms'] for x in m]):.1f}"])
table(rows, ["policy", "steps", "tok/s", "mean TTFT ms", "mean E2E ms"])
explain("Static batching admits 4 requests and waits for the LONGEST to finish before admitting more: finished "
        "slots sit idle. Continuous (iteration-level) batching refills a slot the moment a sequence ends, so the "
        "same work takes fewer steps and later requests start sooner.")

header("2. PagedAttention: memory is allocated in blocks, on demand")
eng = LLMEngine(model, tok, EngineConfig(block_size=16, num_kv_blocks=64))
for p in prompts[:3]:
    eng.add_request(p, SamplingParams(max_tokens=40))
for _ in range(20):
    eng.step()
for s in eng.scheduler.running:
    print(f"  {s.seq_id}: {s.num_tokens:3d} tokens -> blocks {s.block_table}")
print(f"  pool: {eng.bm.num_blocks - eng.bm.num_free_blocks}/{eng.bm.num_blocks} blocks in use")
explain("Each sequence's block table maps logical blocks to arbitrary physical blocks, like virtual memory pages. "
        "Nothing is reserved for tokens not yet generated, so waste is < 1 block per sequence instead of the "
        "60-80% a max-length contiguous reservation wastes.")

header("3. Memory pressure -> preemption (recompute)")
eng, wall, _ = run("continuous", max_num_seqs=8, num_kv_blocks=12, block_size=8, max_num_batched_tokens=32)
print(f"  12 blocks x 8 tokens for 8 requests: {eng.totals['preemptions']} preemptions, all finished: "
      f"{eng.totals['finished'] == len(prompts)}")
explain("When the pool runs out, the most recently admitted sequence is evicted and re-queued; its KV is recomputed "
        "later (often partly from the prefix cache). Correctness is unaffected - only latency.")
