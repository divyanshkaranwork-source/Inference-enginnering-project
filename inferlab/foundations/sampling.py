"""Turning logits into the next token.

Pipeline (same order as vLLM / llama.cpp):

    logits -> repetition / presence / frequency penalties
           -> temperature         (T -> 0 means greedy argmax)
           -> top-k               (keep the k most likely tokens)
           -> top-p  (nucleus)    (keep the smallest set with cumulative prob >= p)
           -> min-p               (drop tokens with prob < min_p * max_prob)
           -> softmax -> sample with a per-request seeded RNG

Sampling runs once per sequence per step on a [vocab] vector: cheap compared
with the forward pass, but engines still fuse it into a GPU kernel because
copying 32k-128k logits per sequence back to the CPU is wasteful.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from inferlab.foundations.ops import log_softmax, softmax

MAX_STOP_SEQUENCES = 4
MAX_STOP_LEN = 64


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0  # 0 disables
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0  # >1 discourages repeats (CTRL-style)
    presence_penalty: float = 0.0  # OpenAI-style additive penalties
    frequency_penalty: float = 0.0
    max_tokens: int = 64
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    logprobs: int = 0  # number of top alternatives to return per token (0..5)
    ignore_eos: bool = False

    def validate(self) -> SamplingParams:
        if not 0.0 <= self.temperature <= 5.0:
            raise ValueError("temperature must be in [0, 5]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not 0.0 <= self.min_p < 1.0:
            raise ValueError("min_p must be in [0, 1)")
        if not 0.5 <= self.repetition_penalty <= 2.0:
            raise ValueError("repetition_penalty must be in [0.5, 2]")
        if not -2.0 <= self.presence_penalty <= 2.0 or not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("presence/frequency penalty must be in [-2, 2]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if len(self.stop) > MAX_STOP_SEQUENCES or any(
            not s or len(s) > MAX_STOP_LEN for s in self.stop
        ):
            raise ValueError(f"at most {MAX_STOP_SEQUENCES} stop strings of 1..{MAX_STOP_LEN} chars")
        if not 0 <= self.logprobs <= 5:
            raise ValueError("logprobs must be in [0, 5]")
        return self

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


# ---------------------------------------------------------------------------
# Individual logit processors (exposed separately for the UI explorer)
# ---------------------------------------------------------------------------
def apply_penalties(
    logits: np.ndarray,
    prev_tokens: list[int] | np.ndarray,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
) -> np.ndarray:
    if len(prev_tokens) == 0 or (
        repetition_penalty == 1.0 and presence_penalty == 0.0 and frequency_penalty == 0.0
    ):
        return logits
    out = logits.copy()
    ids, counts = np.unique(np.asarray(prev_tokens, dtype=np.int64), return_counts=True)
    ids_ok = (ids >= 0) & (ids < out.shape[-1])
    ids, counts = ids[ids_ok], counts[ids_ok]
    if repetition_penalty != 1.0:
        sel = out[ids]
        out[ids] = np.where(sel > 0, sel / repetition_penalty, sel * repetition_penalty)
    out[ids] -= presence_penalty + frequency_penalty * counts
    return out


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = np.partition(logits, -k)[-k]
    return np.where(logits >= kth, logits, -np.inf)


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    if p >= 1.0:
        return logits
    order = np.argsort(-logits, kind="stable")
    probs = softmax(logits[order])
    cum = np.cumsum(probs)
    # keep tokens whose cumulative mass *before* them is < p (always keeps the top token)
    keep_sorted = (cum - probs) < p
    keep = np.zeros_like(keep_sorted)
    keep[order] = keep_sorted
    return np.where(keep, logits, -np.inf)


def min_p_filter(logits: np.ndarray, min_p: float) -> np.ndarray:
    if min_p <= 0.0:
        return logits
    probs = softmax(logits)
    return np.where(probs >= min_p * probs.max(), logits, -np.inf)


def process_logits(logits: np.ndarray, params: SamplingParams, prev_tokens) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = apply_penalties(
        x, prev_tokens, params.repetition_penalty, params.presence_penalty, params.frequency_penalty
    )
    if params.greedy:
        return x
    x = x / params.temperature
    x = top_k_filter(x, params.top_k)
    x = top_p_filter(x, params.top_p)
    x = min_p_filter(x, params.min_p)
    return x


@dataclass
class SampleResult:
    token: int
    logprob: float
    top_logprobs: list[tuple[int, float]]


class Sampler:
    """Stateful sampler for one sequence (owns the seeded RNG)."""

    def __init__(self, params: SamplingParams):
        self.params = params.validate()
        # numpy's PCG64 generator: reproducible per-request when seed is given.
        # (Not a security RNG - sampling needs statistical quality, not secrecy.)
        self.rng = np.random.default_rng(params.seed)

    def sample(self, logits: np.ndarray, prev_tokens, banned: np.ndarray | None = None) -> SampleResult:
        x = process_logits(logits, self.params, prev_tokens)
        if banned is not None:
            x = np.where(banned, -np.inf, x)
        if self.params.greedy:
            token = int(np.argmax(x))
        else:
            probs = softmax(x)
            token = int(self.rng.choice(probs.shape[-1], p=probs))
        # logprobs are reported from the *unprocessed* distribution (OpenAI semantics)
        lp = log_softmax(np.asarray(logits, dtype=np.float64))
        top: list[tuple[int, float]] = []
        if self.params.logprobs:
            idx = np.argpartition(-lp, self.params.logprobs)[: self.params.logprobs]
            idx = idx[np.argsort(-lp[idx])]
            top = [(int(i), float(lp[i])) for i in idx]
        return SampleResult(token, float(lp[token]), top)


def distribution_report(logits: np.ndarray, params: SamplingParams, prev_tokens=(), top: int = 20):
    """Raw vs processed probabilities of the most likely tokens (for the UI)."""
    raw = softmax(np.asarray(logits, dtype=np.float64))
    proc_logits = process_logits(logits, params, list(prev_tokens))
    if params.greedy:
        proc = np.zeros_like(raw)
        proc[int(np.argmax(proc_logits))] = 1.0
    else:
        proc = softmax(proc_logits)
    order = np.argsort(-raw)[:top]
    entropy = float(-np.sum(raw * np.log(np.clip(raw, 1e-30, 1.0))))
    return {
        "tokens": [int(i) for i in order],
        "raw_probs": [float(raw[i]) for i in order],
        "processed_probs": [float(proc[i]) for i in order],
        "kept_tokens": int(np.sum(proc > 0)),
        "entropy_nats": entropy,
    }
