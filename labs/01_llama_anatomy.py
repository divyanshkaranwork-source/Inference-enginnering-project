"""Lab 01 - Anatomy of Llama-2: where the parameters and the time go.

Run:  python labs/01_llama_anatomy.py
Read: docs/01-foundations.md (sections 1-3)
"""

import time

import numpy as np
from _common import explain, header, load, table

from inferlab.config import LLAMA2_7B

model, tok = load()
cfg = model.cfg

header("1. Parameter breakdown: live model vs Llama-2-7B")
a, b = cfg.param_breakdown(), LLAMA2_7B.param_breakdown()
table([[k, f"{a[k] / 1e6:.2f} M", f"{a[k] / cfg.n_params:.0%}", f"{b[k] / 1e9:.3f} B", f"{b[k] / LLAMA2_7B.n_params:.0%}"]
       for k in a], ["component", cfg.name, "share", "llama-2-7b", "share"])
explain("At 7B scale the FFN (SwiGLU: W1, W2, W3) holds ~2/3 of the weights and attention ~1/3. In tiny "
        "models the embedding table dominates. Every decode step multiplies by ALL non-embedding weights once.")

header("2. Shapes of one layer")
lw = model.w.layers[0]
for name in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
    m = getattr(lw, name)
    print(f"  {name}: [{m.out_features:5d} x {m.in_features:5d}]")
explain(f"head_dim = {cfg.dim}/{cfg.n_heads} = {cfg.head_dim}. With {cfg.n_kv_heads} KV heads for {cfg.n_heads} "
        "query heads (GQA group = " + str(cfg.gqa_group) + "), Wk and Wv are " + str(cfg.gqa_group) + "x smaller than Wq.")

header("3. Where does the time go? prefill vs decode")
ids = tok.encode("Once upon a time, there was a little girl who loved to read books about the stars." * 2)
ids = ids[: cfg.max_seq_len - 2]
model.profiler.enabled = True
cache = model.new_kv_cache(len(ids) + 1)
t0 = time.perf_counter()
logits = model.prefill(np.array(ids), cache)
t_pre = time.perf_counter() - t0
pre = model.profiler.summary()
model.profiler.reset()
t0 = time.perf_counter()
model.forward(np.array([int(np.argmax(logits[-1]))]), cache.context(1))
t_dec = time.perf_counter() - t0
dec = model.profiler.summary()
model.profiler.enabled = False
table([[k, f"{pre.get(k, 0) * 1e3:.2f}", f"{dec.get(k, 0) * 1e3:.3f}"] for k in pre],
      ["op", f"prefill {len(ids)} tok (ms)", "decode 1 tok (ms)"])
print(f"\n  prefill: {t_pre * 1e3:.1f} ms total = {t_pre / len(ids) * 1e3:.3f} ms/token")
print(f"  decode : {t_dec * 1e3:.2f} ms for ONE token")
explain("Prefill processes many tokens per weight read (matrix-matrix, compute-bound), so its per-token cost is "
        "far lower than decode, which reads every weight to produce a single token (matrix-vector, memory-bound). "
        "This asymmetry drives every design decision in the runtime layer.")
