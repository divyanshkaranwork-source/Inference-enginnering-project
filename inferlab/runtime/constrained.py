"""Constrained (structured) decoding - force the output into an allowed set.

SGLang, vLLM (via xgrammar / outlines) and TensorRT-LLM can constrain
generation to a regex, JSON schema or grammar. The mechanism is always the
same: before sampling, compute a mask of tokens that keep the output a valid
prefix of *some* allowed string, and set every other logit to -inf.

``ChoiceConstraint`` implements the simplest useful case - "answer must be
one of these strings" (vLLM's ``guided_choice``). Production systems compile
grammars into token-level automata once and cache masks, because scanning the
whole vocabulary per step (as done here) costs O(vocab) Python work.
"""

from __future__ import annotations

import numpy as np

from inferlab.foundations.tokenizer import BaseTokenizer

MAX_CHOICES = 16
MAX_CHOICE_LEN = 128


class ChoiceConstraint:
    def __init__(self, tokenizer: BaseTokenizer, choices: list[str]):
        if not choices or len(choices) > MAX_CHOICES:
            raise ValueError(f"guided_choice needs 1..{MAX_CHOICES} options")
        if any(not c or len(c) > MAX_CHOICE_LEN for c in choices):
            raise ValueError(f"each choice must be 1..{MAX_CHOICE_LEN} characters")
        self.tok = tokenizer
        self.choices = [c.encode("utf-8") for c in choices]
        self.generated = b""

    def banned_mask(self, prev_token: int | None) -> np.ndarray:
        """True = token not allowed at this step."""
        V = self.tok.vocab_size
        banned = np.ones(V, dtype=bool)
        remaining = [c[len(self.generated) :] for c in self.choices if c.startswith(self.generated)]
        complete = any(r == b"" for r in remaining)
        remaining = [r for r in remaining if r]
        for t in range(V):
            if t in self.tok.special_ids:
                continue
            piece = self.tok.token_bytes(t, prev_token)
            if piece and any(r.startswith(piece) for r in remaining):
                banned[t] = False
        if complete:
            banned[self.tok.eos_id] = False
        if banned.all():  # nothing fits: allow EOS so generation terminates
            banned[self.tok.eos_id] = False
        return banned

    def advance(self, token: int, prev_token: int | None) -> None:
        self.generated += self.tok.token_bytes(token, prev_token)

    @property
    def done(self) -> bool:
        return self.generated in self.choices and not any(
            c.startswith(self.generated) and c != self.generated for c in self.choices
        )
