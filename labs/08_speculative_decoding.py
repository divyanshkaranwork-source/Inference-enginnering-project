"""Lab 08 - Speculative decoding.

Run:  python labs/08_speculative_decoding.py
      (with stories15M + stories110M downloaded, set DRAFT=models/stories15M.bin TARGET=models/stories110M.bin)
Read: docs/02-runtime.md (section 6)
"""

import os

from _common import explain, header, load, table

from inferlab.foundations.generate import generate_text
from inferlab.foundations.sampling import SamplingParams
from inferlab.runtime.speculative import speculative_generate

target, tok = load(os.environ.get("TARGET"))
draft = load(os.environ["DRAFT"])[0] if os.environ.get("DRAFT") else None
prompt = "Once upon a time, there was a little girl named Lily. Lily liked to play outside. One day, Lily"
sp = SamplingParams(temperature=0, max_tokens=64)
base_text, base = generate_text(target, tok, prompt, sp)

header("1. Greedy speculative decoding gives exactly the same text")
rows = []
configs = [("ngram k=4", "ngram", None, 4)]
configs += [(f"self-draft {n} layers", "draft_model", target.truncated(n), 4)
            for n in sorted({1, max(1, target.cfg.n_layers // 2)})]
if draft is not None:
    configs.append((f"draft model {draft.cfg.name}", "draft_model", draft, 4))
for name, method, d, k in configs:
    out, st = speculative_generate(target, tok, prompt, sp, method, d, k)
    s = st.to_dict()
    rows.append([name, f"{s['acceptance_rate']:.0%}", f"{s['tokens_per_target_forward']:.2f}", s["target_forwards"],
                 f"{base.total_s / st.total_s:.2f}x", tok.decode(out) == base_text])
table(rows, ["drafter", "accept", "tok/target pass", "target passes", "speedup", "identical"])
print(f"  baseline: {base.completion_tokens} target passes")
explain("Every round costs one target pass and yields accepted+1 tokens. Wall-clock speedup on a CPU is modest "
        "because verifying k tokens is not free here; on a GPU, decode is memory-bound, so verifying 5 tokens "
        "costs almost the same as 1 and speedups of 2-3x are typical (EAGLE-3, Medusa, MTP).")

header("2. With sampling, the output DISTRIBUTION is preserved")
explain("Accept draft token d with probability min(1, p(d)/q(d)); on rejection resample from max(0, p - q). "
        "tests/test_runtime.py::test_rejection_sampling_preserves_distribution checks this empirically.")
