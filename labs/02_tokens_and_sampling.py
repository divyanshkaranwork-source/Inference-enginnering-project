"""Lab 02 - Tokenization and sampling.

Run:  python labs/02_tokens_and_sampling.py
Read: docs/01-foundations.md (sections 4-5)
"""

import numpy as np
from _common import explain, header, load, table

from inferlab.foundations.generate import generate_text
from inferlab.foundations.sampling import SamplingParams, distribution_report

model, tok = load()

header("1. Text -> tokens")
for text in ["Hello world", "Inference engineering", "naïve café 🚀", "</s> is not a control token here"]:
    ids = tok.encode(text)
    print(f"  {text!r:42} -> {len(ids):2d} tokens: {[tok.id_to_piece(i) for i in ids][:10]}")
explain("Unknown characters fall back to raw UTF-8 bytes (<0xF0>...). Special tokens like </s> are only ever "
        "inserted by id, never produced from user text - otherwise users could forge turn boundaries.")

header("2. One forward pass -> a distribution over the vocabulary")
prompt = "Once upon a time, there was a little"
ids = tok.encode(prompt)
logits = model.logits_full(ids)[-1]
for t in (0.0, 0.7, 1.0, 1.5):
    r = distribution_report(logits, SamplingParams(temperature=t), top=5)
    print(f"  T={t:<4} top-5: " + ", ".join(f"{tok.id_to_piece(i)!r} {p:.0%}" for i, p in zip(r["tokens"], r["processed_probs"], strict=True)))
explain("Temperature rescales logits before the softmax: low T sharpens towards argmax, high T flattens. "
        "Top-k / top-p then cut the long tail of implausible tokens.")

header("3. Truncation strategies")
rows = []
for name, sp in [("none", SamplingParams()), ("top_k=10", SamplingParams(top_k=10)),
                 ("top_p=0.9", SamplingParams(top_p=0.9)), ("min_p=0.05", SamplingParams(min_p=0.05))]:
    rows.append([name, distribution_report(logits, sp)["kept_tokens"]])
table(rows, ["filter", "tokens kept"])

header("4. Seeds make sampling reproducible (per request)")
for seed in (1, 1, 2):
    text, _ = generate_text(model, tok, prompt, SamplingParams(temperature=0.9, max_tokens=12, seed=seed))
    print(f"  seed={seed}: {text!r}")
explain("Production servers give every request its own RNG so concurrent users never perturb each other's "
        "randomness - and a fixed seed replays exactly.")
print(f"\n  greedy is deterministic: {np.array_equal(logits.argmax(), logits.argmax())}")
