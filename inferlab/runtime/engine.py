"""LLMEngine - the heart of an inference server (modelled on vLLM's engine).

    add_request()  ->  waiting queue
    step():
        1. scheduler.schedule()        pick sequences + token counts (budgets)
        2. build flat batch            tokens, positions, slot mapping, block tables
        3. model.forward()             ONE batched forward pass over all of them
        4. sample                      one token per sequence that finished its prompt
        5. update / detokenize / stop  stream deltas out, free finished sequences

The ``AsyncEngine`` in ``async_engine.py`` runs ``step()`` in a background
thread so an HTTP server can stream tokens to many clients at once.
"""

from __future__ import annotations

import itertools
import time
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from inferlab.foundations.model import LlamaModel
from inferlab.foundations.sampling import SamplingParams
from inferlab.foundations.tokenizer import BaseTokenizer
from inferlab.runtime.block_manager import BlockManager
from inferlab.runtime.constrained import ChoiceConstraint
from inferlab.runtime.paged import PagedForwardContext, PagedKVPool, SeqAttnMeta
from inferlab.runtime.scheduler import Scheduler, SchedulerConfig
from inferlab.runtime.sequence import SeqStatus, Sequence


@dataclass
class EngineConfig:
    block_size: int = 16
    num_kv_blocks: int = 256
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 256
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    policy: str = "continuous"
    max_model_len: int | None = None  # defaults to the model's context length
    history: int = 200  # step stats kept for the UI


@dataclass
class RequestOutput:
    request_id: str
    token_id: int | None
    text: str  # newly visible text (delta)
    logprob: float | None
    top_logprobs: list[tuple[int, float]]
    finished: bool
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    metrics: dict = field(default_factory=dict)


@dataclass
class StepStats:
    step: int
    t: float
    num_seqs: int
    prefill_tokens: int
    decode_tokens: int
    step_ms: float
    kv_usage: float
    num_waiting: int
    num_running: int
    num_preempted: int


class LLMEngine:
    def __init__(self, model: LlamaModel, tokenizer: BaseTokenizer, cfg: EngineConfig | None = None):
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg or EngineConfig()
        if tokenizer.vocab_size != model.cfg.vocab_size:
            raise ValueError("tokenizer and model vocabulary sizes differ")
        self.max_model_len = min(self.cfg.max_model_len or model.cfg.max_seq_len, model.cfg.max_seq_len)
        self.pool = PagedKVPool(model.cfg, self.cfg.num_kv_blocks, self.cfg.block_size)
        self.bm = BlockManager(self.cfg.num_kv_blocks, self.cfg.block_size, self.cfg.enable_prefix_caching)
        self.scheduler = Scheduler(
            SchedulerConfig(
                self.cfg.max_num_seqs,
                self.cfg.max_num_batched_tokens,
                self.cfg.enable_chunked_prefill,
                self.cfg.policy,
            ),
            self.bm,
        )
        self.seqs: dict[str, Sequence] = {}
        self._ids = itertools.count()
        self.step_count = 0
        self.history: deque[StepStats] = deque(maxlen=self.cfg.history)
        self.totals = {"prompt_tokens": 0, "generation_tokens": 0, "finished": 0, "preemptions": 0}
        self.recent_finished: deque[dict] = deque(maxlen=50)

    # ------------------------------------------------------------------
    def add_request(
        self,
        prompt: str | list[int],
        params: SamplingParams | None = None,
        request_id: str | None = None,
        cache_salt: str | None = None,
        guided_choice: list[str] | None = None,
        arrival_time: float | None = None,
    ) -> Sequence:
        # copy: we clamp max_tokens below and must not mutate the caller's object
        params = replace(params or SamplingParams(), stop=list(params.stop if params else []))
        params.validate()
        ids = self.tok.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if not ids:
            raise ValueError("empty prompt")
        if any(not 0 <= t < self.tok.vocab_size for t in ids):
            raise ValueError("prompt token id out of range")
        if len(ids) >= self.max_model_len:
            raise ValueError(f"prompt is {len(ids)} tokens; max model length is {self.max_model_len}")
        # clamp generation so prompt + output never exceeds the context window
        params.max_tokens = min(params.max_tokens, self.max_model_len - len(ids))
        rid = request_id or f"req-{next(self._ids)}"
        if rid in self.seqs:
            raise ValueError(f"duplicate request id {rid}")
        constraint = ChoiceConstraint(self.tok, guided_choice) if guided_choice else None
        seq = Sequence(rid, ids, params, self.tok.stream_decoder(), cache_salt, constraint, arrival_time)
        self.scheduler.add(seq)
        self.seqs[rid] = seq
        return seq

    def abort(self, request_id: str) -> bool:
        seq = self.seqs.pop(request_id, None)
        if seq is None:
            return False
        self.scheduler.remove(seq)
        seq.status = SeqStatus.FINISHED_ABORTED
        return True

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    # ------------------------------------------------------------------
    def step(self) -> list[RequestOutput]:
        t0 = time.perf_counter()
        sched = self.scheduler.schedule()
        self.totals["preemptions"] += len(sched.preempted)
        if sched.empty:
            return []

        # ---- build the flat batch --------------------------------------
        tokens, metas, row = [], [], 0
        for s in sched.scheduled:
            seq, n = s.seq, s.num_tokens
            start = seq.num_computed_tokens
            tokens.extend(seq.all_tokens[start : start + n])
            metas.append(
                SeqAttnMeta(
                    query_start=row,
                    query_len=n,
                    context_len=start,
                    block_table=np.asarray(seq.block_table, dtype=np.int32),
                    needs_logits=start + n == seq.num_tokens,
                )
            )
            row += n
        ctx = PagedForwardContext(self.pool, metas, self.model.kernels)
        logits = self.model.forward(np.asarray(tokens, dtype=np.int64), ctx)

        # ---- sample + bookkeeping --------------------------------------
        outputs: list[RequestOutput] = []
        prefill_toks = decode_toks = 0
        li = 0
        now = time.monotonic()
        for s, meta in zip(sched.scheduled, metas, strict=True):
            seq = s.seq
            if seq.is_prefill:
                prefill_toks += s.num_tokens
            else:
                decode_toks += s.num_tokens
            seq.num_computed_tokens += s.num_tokens
            self.bm.cache_full_blocks(seq)
            if not meta.needs_logits:
                continue  # mid-prompt chunk: nothing to sample yet
            outputs.append(self._sample_and_update(seq, logits[li], now))
            li += 1

        self.step_count += 1
        self.history.append(
            StepStats(
                step=self.step_count,
                t=time.time(),
                num_seqs=len(sched.scheduled),
                prefill_tokens=prefill_toks,
                decode_tokens=decode_toks,
                step_ms=(time.perf_counter() - t0) * 1e3,
                kv_usage=self.bm.usage(),
                num_waiting=len(self.scheduler.waiting),
                num_running=len(self.scheduler.running),
                num_preempted=len(sched.preempted),
            )
        )
        return outputs

    def _sample_and_update(self, seq: Sequence, logits: np.ndarray, now: float) -> RequestOutput:
        prev = seq.all_tokens[-1]
        banned = seq.constraint.banned_mask(prev) if seq.constraint else None
        res = seq.sampler.sample(logits, seq.output_tokens, banned)
        if seq.constraint:
            seq.constraint.advance(res.token, prev)
        seq.output_tokens.append(res.token)
        seq.logprobs.append(res.logprob)
        if seq.metrics.first_token_time is None:
            seq.metrics.first_token_time = now
            self.totals["prompt_tokens"] += len(seq.prompt_tokens)
        seq.metrics.last_token_time = now
        seq.metrics.token_times.append(now)
        self.totals["generation_tokens"] += 1

        # detokenize incrementally and check stop conditions
        seq.output_text += seq.decoder.feed(res.token)
        finish = None
        p = seq.params
        if res.token == self.tok.eos_id and not p.ignore_eos or seq.constraint is not None and seq.constraint.done:
            finish = "stop"
        elif len(seq.output_tokens) >= p.max_tokens or seq.num_tokens >= self.max_model_len:
            finish = "length"
        stop_idx = None
        for stop in p.stop:
            idx = seq.output_text.find(stop, max(0, seq.sent_text_len - len(stop)))
            if idx != -1 and (stop_idx is None or idx < stop_idx):
                stop_idx = idx
        if stop_idx is not None:  # stop string found: cut the text before it
            finish = "stop"
            visible_end = stop_idx
        elif finish is not None:  # finished for another reason: release everything
            seq.output_text += seq.decoder.flush()
            visible_end = len(seq.output_text)
        elif p.stop:  # hold back text that could be the start of a stop string
            visible_end = len(seq.output_text) - (max(len(x) for x in p.stop) - 1)
        else:
            visible_end = len(seq.output_text)
        visible_end = max(visible_end, seq.sent_text_len)
        delta = seq.output_text[seq.sent_text_len : visible_end]
        seq.sent_text_len = visible_end
        if finish is not None:
            seq.output_text = seq.output_text[:visible_end]
            self._finish(seq, finish)
        return RequestOutput(
            request_id=seq.seq_id,
            token_id=res.token,
            text=delta,
            logprob=res.logprob,
            top_logprobs=res.top_logprobs,
            finished=finish is not None,
            finish_reason=finish,
            prompt_tokens=len(seq.prompt_tokens),
            completion_tokens=len(seq.output_tokens),
            metrics=seq.metrics.to_dict() if finish else {},
        )

    def _finish(self, seq: Sequence, reason: str) -> None:
        seq.status = SeqStatus.FINISHED_STOPPED if reason == "stop" else SeqStatus.FINISHED_LENGTH
        seq.metrics.finished_time = time.monotonic()
        self.scheduler.remove(seq)
        self.seqs.pop(seq.seq_id, None)
        self.totals["finished"] += 1
        self.recent_finished.append(
            {"id": seq.seq_id, "reason": reason, "tokens": len(seq.output_tokens), **seq.metrics.to_dict()}
        )

    # ------------------------------------------------------------------
    def generate(
        self, prompts: list[str | list[int]], params: SamplingParams | list[SamplingParams] | None = None
    ) -> list[dict]:
        """Offline batch API (like ``vllm.LLM.generate``): run all prompts to completion."""
        plist = params if isinstance(params, list) else [params] * len(prompts)
        seqs = [
            self.add_request(p, sp) for p, sp in zip(prompts, plist, strict=True)
        ]
        texts = {s.seq_id: "" for s in seqs}
        while self.has_unfinished():
            for out in self.step():
                if out.request_id in texts:
                    texts[out.request_id] += out.text
        return [
            {
                "request_id": s.seq_id,
                "text": texts[s.seq_id],
                "token_ids": list(s.output_tokens),
                "finish_reason": s.status.value,
                "metrics": s.metrics.to_dict(),
            }
            for s in seqs
        ]

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Everything the UI needs to draw the live engine view."""
        owners: dict[int, list[str]] = {}
        for seq in self.scheduler.running:
            for b in seq.block_table:
                owners.setdefault(b, []).append(seq.seq_id)

        def seq_info(s: Sequence) -> dict:
            return {
                "id": s.seq_id,
                "status": s.status.value,
                "prompt_tokens": len(s.prompt_tokens),
                "output_tokens": len(s.output_tokens),
                "computed_tokens": s.num_computed_tokens,
                "phase": "prefill" if s.is_prefill else "decode",
                "blocks": list(s.block_table),
                "cached_prompt_tokens": s.metrics.num_cached_prompt_tokens,
                "preemptions": s.metrics.num_preemptions,
            }

        return {
            "config": {
                **vars(self.cfg),
                "model": self.model.cfg.name,
                "kv_pool_mb": self.pool.nbytes / 2**20,
                "max_model_len": self.max_model_len,
            },
            "step": self.step_count,
            "running": [seq_info(s) for s in self.scheduler.running],
            "waiting": [seq_info(s) for s in list(self.scheduler.waiting)[:32]],
            "num_waiting": len(self.scheduler.waiting),
            "blocks": self.bm.snapshot(owners),
            "block_stats": self.bm.stats(),
            "history": [vars(h) for h in self.history],
            "totals": dict(self.totals),
            "recent_finished": list(self.recent_finished)[-10:],
        }
