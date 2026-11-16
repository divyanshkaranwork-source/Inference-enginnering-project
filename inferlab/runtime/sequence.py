"""Per-request state tracked by the engine."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

from inferlab.foundations.sampling import Sampler, SamplingParams
from inferlab.foundations.tokenizer import StreamDecoder


class SeqStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED_STOPPED = "stop"
    FINISHED_LENGTH = "length"
    FINISHED_ABORTED = "abort"

    @property
    def finished(self) -> bool:
        return self.value in ("stop", "length", "abort")


@dataclass
class SeqMetrics:
    arrival_time: float
    first_scheduled_time: float | None = None
    first_token_time: float | None = None
    last_token_time: float | None = None
    finished_time: float | None = None
    num_preemptions: int = 0
    num_cached_prompt_tokens: int = 0  # prefix-cache hits
    token_times: list[float] = field(default_factory=list)

    @property
    def queue_time(self) -> float | None:
        if self.first_scheduled_time is None:
            return None
        return self.first_scheduled_time - self.arrival_time

    @property
    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    def to_dict(self) -> dict:
        t = self.token_times
        itl = [b - a for a, b in zip(t, t[1:], strict=False)]
        return {
            "queue_ms": None if self.queue_time is None else self.queue_time * 1e3,
            "ttft_ms": None if self.ttft is None else self.ttft * 1e3,
            "tpot_ms": (sum(itl) / len(itl) * 1e3) if itl else None,
            "e2e_ms": None
            if self.finished_time is None
            else (self.finished_time - self.arrival_time) * 1e3,
            "num_preemptions": self.num_preemptions,
            "cached_prompt_tokens": self.num_cached_prompt_tokens,
        }


class Sequence:
    """One request: its tokens, KV blocks and sampling state.

    Invariant used by the scheduler (same as vLLM v1):
        tokens to compute this step = num_tokens - num_computed_tokens
    A fresh request has num_computed_tokens = 0 (prefill); a request in
    decode has exactly one uncomputed token: the one just sampled.
    """

    def __init__(
        self,
        seq_id: str,
        prompt_tokens: list[int],
        params: SamplingParams,
        decoder: StreamDecoder,
        cache_salt: str | None = None,
        constraint=None,
        arrival_time: float | None = None,
    ):
        if not prompt_tokens:
            raise ValueError("prompt must contain at least one token")
        self.seq_id = seq_id
        self.prompt_tokens = list(prompt_tokens)
        self.output_tokens: list[int] = []
        self.params = params
        self.sampler = Sampler(params)
        self.decoder = decoder
        self.cache_salt = cache_salt
        self.constraint = constraint
        self.status = SeqStatus.WAITING
        self.num_computed_tokens = 0
        self.block_table: list[int] = []
        self.block_hashes: list[bytes] = []  # hashes of full blocks (prefix caching)
        self.output_text = ""
        self.sent_text_len = 0
        self.logprobs: list[float] = []
        self.metrics = SeqMetrics(arrival_time if arrival_time is not None else time.monotonic())

    # ------------------------------------------------------------------
    @property
    def num_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def all_tokens(self) -> list[int]:
        return self.prompt_tokens + self.output_tokens

    @property
    def num_uncomputed(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens < len(self.prompt_tokens)

    @property
    def finished(self) -> bool:
        return self.status.finished

    def max_total_tokens(self) -> int:
        return len(self.prompt_tokens) + self.params.max_tokens

    def __repr__(self) -> str:
        return (
            f"Sequence({self.seq_id}, {self.status.value}, tokens={self.num_tokens}, "
            f"computed={self.num_computed_tokens}, blocks={self.block_table})"
        )
