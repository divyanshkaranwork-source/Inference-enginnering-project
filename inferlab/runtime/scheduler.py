"""Iteration-level scheduling: continuous batching with chunked prefill.

Every engine step the scheduler decides *which sequences run and how many of
their tokens* are processed, under two budgets:

* ``max_num_batched_tokens`` - compute budget per step (keeps step latency
  bounded so decoding users do not stall behind a huge prompt),
* KV-cache blocks             - memory budget.

Policies

* ``continuous`` (Orca / vLLM): finished sequences leave and waiting ones join
  at *every* step. Long prompts are split into chunks (chunked prefill) and
  mixed with decode tokens of other sequences.
* ``static`` (naive batching): a batch is admitted, then runs until *all* of
  its sequences finish. Short requests wait for the longest one - head-of-line
  blocking. Lab 06 compares the two.

When memory runs out, the most recently admitted running sequence is
*preempted*: its blocks are freed and it goes back to the front of the waiting
queue to be recomputed later (vLLM's "recompute" preemption). With prefix
caching on, much of that recompute is a cache hit.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from inferlab.runtime.block_manager import BlockManager
from inferlab.runtime.sequence import SeqStatus, Sequence


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 256
    enable_chunked_prefill: bool = True
    policy: str = "continuous"  # "continuous" | "static"

    def __post_init__(self) -> None:
        if self.policy not in ("continuous", "static"):
            raise ValueError("policy must be 'continuous' or 'static'")
        if self.max_num_seqs < 1 or self.max_num_batched_tokens < 1:
            raise ValueError("scheduler budgets must be positive")


@dataclass
class ScheduledSeq:
    seq: Sequence
    num_tokens: int

    @property
    def is_prefill(self) -> bool:
        return self.seq.is_prefill


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledSeq] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)

    @property
    def num_batched_tokens(self) -> int:
        return sum(s.num_tokens for s in self.scheduled)

    @property
    def empty(self) -> bool:
        return not self.scheduled


class Scheduler:
    def __init__(self, cfg: SchedulerConfig, block_manager: BlockManager):
        self.cfg = cfg
        self.bm = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self._static_batch_open = True

    def add(self, seq: Sequence) -> None:
        if not self.bm.can_ever_fit(seq.max_total_tokens()):
            raise ValueError("request needs more KV-cache blocks than the whole pool holds")
        if not self.cfg.enable_chunked_prefill and seq.max_total_tokens() > self.cfg.max_num_batched_tokens:
            raise ValueError("without chunked prefill, prompt + max_tokens must fit max_num_batched_tokens")
        self.waiting.append(seq)

    def remove(self, seq: Sequence) -> None:
        if seq in self.running:
            self.running.remove(seq)
        elif seq in self.waiting:
            self.waiting.remove(seq)
        self.bm.free(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    # ------------------------------------------------------------------
    def _preempt(self, seq: Sequence, out: SchedulerOutput) -> None:
        self.running.remove(seq)
        self.bm.free(seq)
        seq.num_computed_tokens = 0
        seq.status = SeqStatus.WAITING
        seq.metrics.num_preemptions += 1
        self.waiting.appendleft(seq)
        out.preempted.append(seq)

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.cfg.max_num_batched_tokens
        chunked = self.cfg.enable_chunked_prefill

        # 1) running sequences first (decode tokens and unfinished prefill chunks)
        i = 0
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            n = seq.num_uncomputed
            if n > budget and not chunked:
                i += 1
                continue
            n = min(n, budget)
            while not self.bm.allocate_slots(seq, n):
                victim = self.running[-1]  # lowest priority = admitted last
                self._preempt(victim, out)
                if victim is seq:
                    break
            else:
                out.scheduled.append(ScheduledSeq(seq, n))
                budget -= n
                i += 1

        # 2) admit waiting sequences
        can_admit = not out.preempted
        if self.cfg.policy == "static":
            if not self.running:
                self._static_batch_open = True
            can_admit = can_admit and self._static_batch_open
        static = self.cfg.policy == "static"
        while can_admit and self.waiting and len(self.running) < self.cfg.max_num_seqs:
            if budget <= 0 and not static:
                break
            seq = self.waiting[0]
            computed_blocks, num_cached = ([], 0)
            if seq.num_computed_tokens == 0:
                computed_blocks, num_cached = self.bm.get_computed_blocks(seq)
            n = seq.num_tokens - num_cached
            if n > budget:
                if static:  # a static batch is formed all at once; compute later
                    n = budget if chunked else 0
                elif not chunked:
                    break
                else:
                    n = budget
            if not self.bm.allocate_slots(seq, n, computed_blocks):
                break  # out of KV memory: wait for running sequences to finish
            self.bm.record_prefix_lookup(len(seq.prompt_tokens), num_cached)
            seq.num_computed_tokens = num_cached
            seq.metrics.num_cached_prompt_tokens = max(seq.metrics.num_cached_prompt_tokens, num_cached)
            if seq.metrics.first_scheduled_time is None:
                seq.metrics.first_scheduled_time = time.monotonic()
            self.waiting.popleft()
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            if n > 0:
                out.scheduled.append(ScheduledSeq(seq, n))
                budget -= n
        if self.cfg.policy == "static" and self.running:
            self._static_batch_open = False  # batch is sealed until it drains
        return out
