"""The simplest possible inference loop - and why it is slow.

``generate`` runs one sequence at a time:

    prefill:  forward(all prompt tokens)          -> first token   (TTFT)
    decode:   forward(1 new token) x max_tokens   -> one token each (TPOT)

With ``use_kv_cache=False`` every step re-runs the *whole* sequence, so the
cost of token n grows with n (O(n^2) total). With the KV cache each decode
step only processes one token. Lab 03 measures the difference.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from inferlab.foundations.model import LlamaModel
from inferlab.foundations.sampling import Sampler, SamplingParams
from inferlab.foundations.tokenizer import BaseTokenizer


@dataclass
class TokenEvent:
    token: int
    text: str
    logprob: float
    top_logprobs: list[tuple[int, float]]
    latency_s: float  # time spent producing this token
    index: int


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_s: float = 0.0
    decode_latencies_s: list[float] = field(default_factory=list)
    total_s: float = 0.0
    finish_reason: str = "length"

    @property
    def tpot_s(self) -> float:
        d = self.decode_latencies_s
        return float(np.mean(d)) if d else 0.0

    @property
    def tokens_per_s(self) -> float:
        return self.completion_tokens / self.total_s if self.total_s else 0.0

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_ms": self.ttft_s * 1e3,
            "tpot_ms": self.tpot_s * 1e3,
            "total_ms": self.total_s * 1e3,
            "tokens_per_s": self.tokens_per_s,
            "finish_reason": self.finish_reason,
            "decode_latencies_ms": [x * 1e3 for x in self.decode_latencies_s],
        }


def generate(
    model: LlamaModel,
    tokenizer: BaseTokenizer,
    prompt: str | list[int],
    params: SamplingParams | None = None,
    use_kv_cache: bool = True,
    stats: GenerationStats | None = None,
) -> Iterator[TokenEvent]:
    params = (params or SamplingParams()).validate()
    stats = stats if stats is not None else GenerationStats()
    ids = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
    max_len = model.cfg.max_seq_len
    if len(ids) >= max_len:
        raise ValueError(f"prompt has {len(ids)} tokens; model context is {max_len}")
    budget = min(params.max_tokens, max_len - len(ids))
    stats.prompt_tokens = len(ids)
    sampler = Sampler(params)
    detok = tokenizer.stream_decoder()
    cache = model.new_kv_cache(len(ids) + budget) if use_kv_cache else None
    tokens = list(ids)
    text_so_far = ""
    t_start = time.perf_counter()

    for i in range(budget):
        t0 = time.perf_counter()
        if cache is None:  # recompute everything: O(n) work per token
            logits = model.logits_full(np.array(tokens))[-1]
        elif i == 0:  # prefill
            logits = model.prefill(np.array(tokens), cache)[-1]
        else:  # decode: only the newest token goes through the network
            logits = model.forward(np.array(tokens[-1:]), cache.context(1, kernels=model.kernels))[-1]
        res = sampler.sample(logits, tokens[len(ids) :])
        dt = time.perf_counter() - t0
        if i == 0:
            stats.ttft_s = dt
        else:
            stats.decode_latencies_s.append(dt)
        tokens.append(res.token)
        stats.completion_tokens += 1
        piece = detok.feed(res.token)
        text_so_far += piece
        yield TokenEvent(res.token, piece, res.logprob, res.top_logprobs, dt, i)
        if res.token == tokenizer.eos_id and not params.ignore_eos:
            stats.finish_reason = "stop"
            break
        if params.stop and any(s in text_so_far for s in params.stop):
            stats.finish_reason = "stop"
            break
    stats.total_s = time.perf_counter() - t_start


def generate_text(model, tokenizer, prompt, params=None, use_kv_cache=True) -> tuple[str, GenerationStats]:
    stats = GenerationStats()
    text = "".join(ev.text for ev in generate(model, tokenizer, prompt, params, use_kv_cache, stats))
    return text, stats
