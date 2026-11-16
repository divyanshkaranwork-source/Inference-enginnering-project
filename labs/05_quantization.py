"""Lab 05 - Weight quantization: fewer bytes, faster decode, (slightly) different answers.

Run:  python labs/05_quantization.py
Read: docs/01-foundations.md (section 8), docs/07-advanced.md (AWQ / GPTQ / FP8)
"""

import numpy as np
from _common import explain, header, load, table

from inferlab.config import LLAMA2_7B
from inferlab.foundations import perfmodel as pm
from inferlab.foundations.generate import generate_text
from inferlab.foundations.model import requantize
from inferlab.foundations.ops import log_softmax, softmax
from inferlab.foundations.sampling import SamplingParams

model, tok = load()
prompt = "Once upon a time, there was a little girl named Lily. She loved"
ids = np.array(tok.encode(prompt))
base = model.logits_full(ids)
p = softmax(base.astype(np.float64))

header("1. Same model, six weight formats")
rows = []
for kind in ("fp32", "fp16", "bf16", "int8", "q8_0", "q4_0"):
    q = model if kind == "fp32" else requantize(model, kind)
    lg = q.logits_full(ids)
    kl = float(np.mean(np.sum(p * (np.log(np.clip(p, 1e-30, 1)) - log_softmax(lg.astype(np.float64))), -1)))
    text, _ = generate_text(q, tok, prompt, SamplingParams(temperature=0, max_tokens=12))
    rows.append([kind, f"{q.nbytes / 2**20:.1f}", f"{kl:.2e}", f"{np.mean(lg.argmax(-1) == base.argmax(-1)):.0%}",
                 repr(text[:40])])
table(rows, ["format", "MiB", "KL vs fp32", "top-1 agree", "greedy continuation"])
explain("fp16/bf16 are near-lossless. INT8 per-channel and Q8_0 (blocks of 32 sharing a scale) are nearly so. "
        "Q4_0 keeps only 16 levels per block: visible but usually acceptable drift - and 7x fewer bytes than fp32.")

header("2. What it means for Llama-2-7B decode on an RTX 4090 (24 GB)")
rows = []
for dt in ("fp16", "int8", "q4_k_m"):
    d = pm.decode_step(LLAMA2_7B, "RTX-4090-24GB", batch=1, context_len=1024, weight_dtype=dt)
    rows.append([dt, f"{pm.weight_bytes(LLAMA2_7B, dt) / 1e9:.1f} GB", f"{d['tokens_per_s']:.0f} tok/s"])
table(rows, ["weights", "size", "batch-1 decode"])
explain("Decode speed tracks bytes read per token, so 4-bit weights roughly triple single-user speed - the reason "
        "llama.cpp's GGUF Q4_K_M is the default way to run 7B models on laptops.")
