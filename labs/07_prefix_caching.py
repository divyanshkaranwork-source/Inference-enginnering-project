"""Lab 07 - Prefix caching: vLLM block hashing vs SGLang RadixAttention.

Run:  python labs/07_prefix_caching.py
Read: docs/02-runtime.md (section 5)
"""

import time

from _common import explain, header, load

from inferlab.foundations.sampling import SamplingParams
from inferlab.runtime.engine import EngineConfig, LLMEngine
from inferlab.runtime.radix_cache import RadixCache

model, tok = load()
system = ("You are a friendly storyteller. Always write short, happy stories for small children, with simple words "
          "and a kind ending. ")
system = system * max(1, (model.cfg.max_seq_len // 3) // len(tok.encode(system)))
questions = ["Tell me about a cat.", "Tell me about a dog.", "Tell me about a bird.", "Tell me about a fish."]

header("1. Shared system prompt with and without prefix caching")
for enabled in (False, True):
    eng = LLMEngine(model, tok, EngineConfig(enable_prefix_caching=enabled, block_size=16))
    t0 = time.perf_counter()
    ttfts = []
    for q in questions:  # sequential requests (like a chat app)
        s = eng.add_request(system + q, SamplingParams(temperature=0, max_tokens=8))
        while eng.has_unfinished():
            eng.step()
        ttfts.append(s.metrics.ttft * 1e3)
    st = eng.bm.stats()
    print(f"  caching={enabled!s:5}  TTFT per request (ms): {[round(x, 1) for x in ttfts]}  "
          f"hit rate {st['prefix_hit_rate']:.0%}  total {(time.perf_counter() - t0) * 1e3:.0f} ms")
explain("With caching on, requests 2-4 reuse the KV blocks of the shared prefix and only prefill their own "
        "question: TTFT drops sharply. vLLM identifies blocks by a hash chain over their tokens (SHA-256 here).")

header("2. Tenant isolation with cache_salt")
eng = LLMEngine(model, tok, EngineConfig(block_size=16))
eng.generate([system + questions[0]], SamplingParams(max_tokens=2))
a = eng.add_request(system + questions[1], SamplingParams(max_tokens=2))
b = eng.add_request(system + questions[2], SamplingParams(max_tokens=2), cache_salt="tenant-B")
while eng.has_unfinished():
    eng.step()
print(f"  same tenant reused {a.metrics.num_cached_prompt_tokens} tokens; other tenant reused "
      f"{b.metrics.num_cached_prompt_tokens}")
explain("A shared cache is a side channel: if my request is suspiciously fast, someone else sent the same prefix. "
        "Salting the hash per tenant removes cross-tenant sharing (vLLM: cache_salt).")

header("3. SGLang's radix tree")
rc = RadixCache()
for q in questions:
    hit = rc.insert(tok.encode(system + q))
    print(f"  insert {q!r:26} -> {hit} tokens already in the tree")
tree = rc.to_dict(decode=tok.decode)["tree"]
print(f"  root children: {len(tree['children'])}, shared edge: {tree['children'][0]['tokens']} tokens, "
      f"branches: {len(tree['children'][0]['children'])}")
explain("A radix tree stores token runs on its edges: the shared system prompt is ONE edge with four branches. "
        "Token-level matching (no block rounding) and LRU eviction of leaves make it ideal for multi-turn chat and "
        "agents.")
