"""Request schemas (OpenAI-compatible) with strict bounds.

Unknown fields are ignored for client compatibility, but every field we act
on is type-checked and range-limited. Dynamic limits (max prompt length,
max_tokens) come from settings and are enforced in the endpoint.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from inferlab.foundations.sampling import SamplingParams


class _GenParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = Field(None, max_length=256)
    max_tokens: int | None = Field(None, ge=1)
    temperature: float = Field(1.0, ge=0.0, le=5.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    top_k: int = Field(0, ge=-1, le=10000)
    min_p: float = Field(0.0, ge=0.0, lt=1.0)
    repetition_penalty: float = Field(1.0, ge=0.5, le=2.0)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    stop: str | list[str] | None = None
    seed: int | None = Field(None, ge=0, le=2**63 - 1)
    stream: bool = False
    n: int = Field(1, ge=1, le=1)
    ignore_eos: bool = False
    guided_choice: list[str] | None = Field(None, min_length=1, max_length=16)
    cache_salt: str | None = Field(None, min_length=1, max_length=128)
    user: str | None = Field(None, max_length=256)

    @field_validator("stop")
    @classmethod
    def _stop(cls, v):
        if v is None:
            return []
        items = [v] if isinstance(v, str) else v
        if len(items) > 4 or any(not s or len(s) > 64 for s in items):
            raise ValueError("stop: at most 4 strings of 1..64 characters")
        return items

    @field_validator("guided_choice")
    @classmethod
    def _choices(cls, v):
        if v is not None and any(not c or len(c) > 128 for c in v):
            raise ValueError("guided_choice: each option must be 1..128 characters")
        return v

    def sampling(self, max_tokens: int, logprobs: int = 0) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            top_k=max(self.top_k, 0),
            top_p=self.top_p,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            max_tokens=max_tokens,
            stop=list(self.stop or []),
            seed=self.seed,
            logprobs=logprobs,
            ignore_eos=self.ignore_eos,
        ).validate()


class CompletionRequest(_GenParams):
    prompt: str | list[int]
    logprobs: int | None = Field(None, ge=0, le=5)
    echo: bool = False

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, v):
        if isinstance(v, list):
            if not v or len(v) > 32768 or any(t < 0 for t in v):
                raise ValueError("token prompt must have 1..32768 non-negative ids")
        elif v == "":
            raise ValueError("prompt must not be empty")
        return v

    @field_validator("echo")
    @classmethod
    def _echo(cls, v):
        if v:
            raise ValueError("echo is not supported")
        return v


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., max_length=1_000_000)


class ChatRequest(_GenParams):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=64)
    max_completion_tokens: int | None = Field(None, ge=1)
    logprobs: bool = False
    top_logprobs: int | None = Field(None, ge=0, le=5)
