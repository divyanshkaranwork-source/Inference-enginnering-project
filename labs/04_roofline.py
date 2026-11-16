"""Lab 04 - Roofline: is a step limited by compute or by memory bandwidth?

Run:  python labs/04_roofline.py
Read: docs/01-foundations.md (section 7)
"""

import time

import numpy as np
from _common import explain, header, table

from inferlab.config import LLAMA2_7B
from inferlab.foundations import perfmodel as pm

header("1. Measure YOUR machine: same weights, growing batch")
K = N = 4096
W = np.random.default_rng(0).standard_normal((N, K), dtype=np.float32)
rows = []
for b in (1, 4, 16, 64, 256):
    x = np.ones((b, K), np.float32)
    x @ W.T
    t0 = time.perf_counter()
    for _ in range(5):
        x @ W.T
    t = (time.perf_counter() - t0) / 5
    flops, bytes_ = 2 * b * N * K, W.nbytes + 2 * x.nbytes
    rows.append([b, f"{t * 1e3:.2f}", f"{flops / t / 1e9:.1f}", f"{flops / bytes_:.1f}", f"{b / t:.0f}"])
table(rows, ["batch", "ms", "GFLOP/s", "FLOP/byte", "rows/s"])
explain("At batch 1 the 64 MiB weight matrix is streamed from memory for just 2 FLOPs per 4 bytes. Increasing the "
        "batch reuses each loaded weight for more rows: time barely grows until arithmetic intensity reaches the "
        "machine's ridge point. Throughput (rows/s) climbs almost linearly - batching is free until then.")

header("2. Llama-2-7B decode on real GPUs (fp16, context 1024)")
rows = []
for gpu in ("L4-24GB", "A100-SXM-80GB", "H100-SXM-80GB", "Apple-M2-Max"):
    for b in (1, 32):
        d = pm.decode_step(LLAMA2_7B, gpu, batch=b, context_len=1024)
        rows.append([gpu, b, f"{d['arithmetic_intensity']:.1f}", d["bound"], f"{d['step_time_ms']:.1f}",
                     f"{d['tokens_per_s']:.0f}"])
table(rows, ["gpu", "batch", "FLOP/B", "bound", "ms/step", "tok/s"])
g = pm.get_gpu("H100-SXM-80GB")
explain(f"An H100 needs ~{g.ridge_point:.0f} FLOP/byte to be compute-bound (989 TFLOPS / 3.35 TB/s). Decode sits "
        "near 1-30: memory-bound. Lower bound for one token at batch 1 = 13.5 GB / 3.35 TB/s = 4 ms -> ~250 tok/s "
        "per user no matter how fast the math is. Quantization (lab 05) lowers the bytes; batching (lab 06) "
        "amortises them across users.")

header("3. Prefill is the opposite")
for n in (128, 2048):
    p = pm.prefill(LLAMA2_7B, "H100-SXM-80GB", prompt_len=n)
    print(f"  prefill {n:5d} tokens: AI={p['arithmetic_intensity']:.0f} FLOP/B -> {p['bound']}-bound, "
          f"TTFT ~ {p['ttft_ms']:.1f} ms")
