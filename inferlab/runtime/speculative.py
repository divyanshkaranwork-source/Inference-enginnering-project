"""Speculative decoding - several tokens per expensive forward pass, same output.

Decode is memory-bound: verifying k+1 tokens in one target forward pass costs
almost the same as generating 1. So we let a cheap *drafter* guess k tokens and
have the target model check them all at once:

    draft:   x -> d1 d2 d3 d4            (cheap: small model or n-gram lookup)
    verify:  target([x, d1, d2, d3, d4]) -> p0 p1 p2 p3 p4   (one pass)
    accept d_i with prob min(1, p(d_i) / q(d_i)); on the first rejection
    resample from norm(max(0, p - q)); if all accepted take a bonus token ~ p4.

This rejection scheme (Leviathan et al. 2023, Chen et al. 2023) provably
leaves the output distribution *identical* to sampling from the target alone.
With greedy decoding it reduces to "accept while draft == argmax".

Drafters implemented:

* ``draft_model`` - a smaller model sharing the tokenizer (e.g. stories15M
  drafting for stories110M), or the first layers of the target itself
  (self-speculation / LayerSkip - no extra weights).
* ``ngram`` - prompt-lookup decoding: find the last n-gram earlier in the
  context and propose what followed it. Free, great for summarisation / code
  editing where output copies the input. (vLLM ``[ngram]``, llama.cpp lookup)

Engines: vLLM (draft, EAGLE, Medusa, MTP, ngram), SGLang (EAGLE-2/3),
TensorRT-LLM (draft-target, Medusa, EAGLE, lookahead), llama.cpp (--model-draft).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from inferlab.foundations.model import LlamaModel
from inferlab.foundations.ops import softmax
from inferlab.foundations.sampling import SamplingParams, process_logits
from inferlab.foundations.tokenizer import BaseTokenizer


def _probs(logits: np.ndarray, params: SamplingParams, prev) -> np.ndarray:
    x = process_logits(logits, params, prev)
    if params.greedy:
        p = np.zeros_like(x)
        p[int(np.argmax(x))] = 1.0
        return p
    return softmax(x)


def verify_draft(
    draft_tokens: list[int],
    q_probs: list[np.ndarray] | None,
    p_probs: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Speculative-sampling acceptance.

    draft_tokens: k proposals; q_probs: drafter distributions (None => the
    drafter was deterministic, q = one-hot); p_probs: [k+1, V] target
    distributions. Returns (num_accepted, next_token).
    """
    k = len(draft_tokens)
    for i, d in enumerate(draft_tokens):
        p = p_probs[i]
        q = q_probs[i] if q_probs is not None else None
        qd = 1.0 if q is None else float(q[d])
        if qd > 0 and rng.random() < min(1.0, float(p[d]) / qd):
            continue  # accepted
        # rejected: resample from the residual distribution max(0, p - q)
        if q is None:
            resid = p.copy()
            resid[d] = 0.0
        else:
            resid = np.maximum(p - q, 0.0)
        s = resid.sum()
        nxt = int(rng.choice(len(p), p=resid / s)) if s > 0 else int(np.argmax(p))
        return i, nxt
    p_last = p_probs[k]
    return k, int(rng.choice(len(p_last), p=p_last))


def ngram_propose(tokens: list[int], k: int, max_n: int = 3, min_n: int = 1) -> list[int]:
    """Prompt-lookup: match the last n tokens earlier in the context."""
    L = len(tokens)
    for n in range(min(max_n, L - 1), min_n - 1, -1):
        tail = tokens[L - n :]
        for start in range(L - n - 1, -1, -1):  # most recent match first
            if tokens[start : start + n] == tail:
                cont = tokens[start + n : start + n + k]
                if cont:
                    return cont
    return []


@dataclass
class SpecStats:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0
    total_s: float = 0.0
    trace: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        n_tok = sum(r["emitted"] for r in self.trace)
        return {
            "rounds": self.rounds,
            "proposed": self.proposed,
            "accepted": self.accepted,
            "acceptance_rate": self.accepted / self.proposed if self.proposed else 0.0,
            "tokens_per_target_forward": n_tok / self.target_forwards if self.target_forwards else 0.0,
            "target_forwards": self.target_forwards,
            "draft_forwards": self.draft_forwards,
            "total_ms": self.total_s * 1e3,
            "tokens_per_s": n_tok / self.total_s if self.total_s else 0.0,
            "trace": self.trace[:200],
        }


def speculative_generate(
    target: LlamaModel,
    tokenizer: BaseTokenizer,
    prompt: str | list[int],
    params: SamplingParams,
    method: str = "ngram",
    draft: LlamaModel | None = None,
    k: int = 4,
) -> tuple[list[int], SpecStats]:
    if method not in ("ngram", "draft_model"):
        raise ValueError("method must be 'ngram' or 'draft_model'")
    if method == "draft_model" and draft is None:
        raise ValueError("draft_model method requires a draft model")
    if not 1 <= k <= 16:
        raise ValueError("k must be in [1, 16]")
    params = params.validate()
    rng = np.random.default_rng(params.seed)
    ids = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
    max_len = target.cfg.max_seq_len
    if draft is not None:
        max_len = min(max_len, draft.cfg.max_seq_len)
    if len(ids) >= max_len:
        raise ValueError("prompt longer than the context window")
    budget = min(params.max_tokens, max_len - len(ids) - 1)
    budget = max(budget, 1)
    st = SpecStats()
    t0 = time.perf_counter()

    tcache = target.new_kv_cache(len(ids) + budget + k + 2)
    dcache = draft.new_kv_cache(len(ids) + budget + k + 2) if method == "draft_model" else None
    out: list[int] = []

    # prefill the target (all but the last prompt token stay "pending")
    tokens = list(ids)
    logits = target.prefill(np.array(tokens), tcache)[-1]
    st.target_forwards += 1
    first = int(
        verify_draft([], None, _probs(logits, params, out)[None], rng)[1]
    )
    tokens.append(first)
    out.append(first)

    while len(out) < budget and not (first == tokenizer.eos_id and not params.ignore_eos):
        if out[-1] == tokenizer.eos_id and not params.ignore_eos:
            break
        kk = min(k, budget - len(out))
        # ---- 1. draft ---------------------------------------------------
        q_probs: list[np.ndarray] | None
        if method == "ngram":
            proposal = ngram_propose(tokens, kk)
            q_probs = None
        else:
            proposal, q_probs = [], []
            dctx = tokens[dcache.length :]  # catch the draft cache up
            for _ in range(kk):
                dl = draft.forward(np.array(dctx), dcache.context(len(dctx)))[-1]
                st.draft_forwards += 1
                q = _probs(dl, params, out + proposal)
                d = int(np.argmax(q)) if params.greedy else int(rng.choice(len(q), p=q))
                proposal.append(d)
                q_probs.append(q)
                dctx = [d]
        # ---- 2. verify all proposals in ONE target forward pass ---------
        verify_in = [tokens[-1]] + proposal
        base_len = tcache.length
        vl = target.forward(np.array(verify_in), tcache.context(len(verify_in), all_logits=True))
        st.target_forwards += 1
        p_probs = np.stack([_probs(vl[i], params, out + proposal[:i]) for i in range(len(verify_in))])
        n_acc, nxt = verify_draft(proposal, q_probs, p_probs, rng)
        # ---- 3. roll back the KV caches past the first rejection --------
        tcache.truncate(base_len + 1 + n_acc)
        if dcache is not None:
            dcache.truncate(min(dcache.length, len(tokens) + n_acc))
        new = proposal[:n_acc] + [nxt]
        st.rounds += 1
        st.proposed += len(proposal)
        st.accepted += n_acc
        st.trace.append(
            {"proposed": proposal, "accepted": n_acc, "emitted": len(new), "tokens": new}
        )
        for t in new:
            if len(out) >= budget:
                break
            tokens.append(t)
            out.append(t)
            if t == tokenizer.eos_id and not params.ignore_eos:
                break
        if out[-1] == tokenizer.eos_id and not params.ignore_eos:
            break
    st.total_s = time.perf_counter() - t0
    return out, st
