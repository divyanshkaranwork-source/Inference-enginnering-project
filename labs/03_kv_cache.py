"""Lab 03 - The KV cache: from O(n^2) to O(n), and what it costs in memory.

Run:  python labs/03_kv_cache.py
Read: docs/01-foundations.md (section 6)
"""

from _common import explain, header, load, table

from inferlab.config import LLAMA2_7B, LLAMA2_70B, LLAMA3_8B
from inferlab.foundations import perfmodel as pm
from inferlab.foundations.generate import generate_text
from inferlab.foundations.sampling import SamplingParams

model, tok = load()
n = min(64, model.cfg.max_seq_len // 2)
sp = SamplingParams(temperature=0, max_tokens=n, ignore_eos=True)

header(f"1. Generate {n} tokens with and without the KV cache")
t1, s1 = generate_text(model, tok, "Once upon a time", sp, use_kv_cache=True)
t2, s2 = generate_text(model, tok, "Once upon a time", sp, use_kv_cache=False)
d1, d2 = s1.decode_latencies_s, s2.decode_latencies_s
table([[i + 2, f"{d1[i] * 1e3:.2f}", f"{d2[i] * 1e3:.2f}"] for i in range(0, len(d1), max(1, len(d1) // 8))],
      ["token #", "with cache (ms)", "without (ms)"])
print(f"\n  total: {s1.total_s * 1e3:.0f} ms vs {s2.total_s * 1e3:.0f} ms  ->  {s2.total_s / s1.total_s:.1f}x faster, "
      f"identical output: {t1 == t2}")
explain("Without a cache, token n re-runs the whole n-token sequence: latency grows linearly per token "
        "(quadratic in total). With the cache, each step only computes the new token's Q/K/V and attends "
        "to stored keys/values.")

header("2. The price: KV-cache memory (fp16)")
rows = []
for cfg in (LLAMA2_7B, LLAMA3_8B, LLAMA2_70B):
    per_tok = pm.kv_bytes_per_token(cfg, "fp16")
    rows.append([cfg.name, f"{cfg.n_heads}/{cfg.n_kv_heads}", f"{per_tok / 1024:.0f} KiB",
                 f"{per_tok * 4096 / 2**30:.2f} GiB", f"{per_tok * 4096 * 32 / 2**30:.0f} GiB"])
table(rows, ["model", "heads/kv", "per token", "4k-token seq", "32 seqs x 4k"])
explain("Llama-2-7B uses full multi-head attention: 512 KiB per token, 2 GiB per 4k-token sequence - 32 users "
        "need 64 GiB just for KV. Grouped-query attention (Llama-3-8B, Llama-2-70B: 8 KV heads) cuts this 4-8x. "
        "Managing this memory well is the job of PagedAttention (lab 06).")
