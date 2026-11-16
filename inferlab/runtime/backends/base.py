"""Backend interface: anything that can stream tokens for a request.

The production gateway (``inferlab.server``) only talks to this interface,
so the same authenticated, rate-limited, metered API can sit in front of

* the built-in NumPy engine (``native``) - runs anywhere, fully observable,
* llama.cpp's ``llama-server``, vLLM, SGLang or TensorRT-LLM's
  ``trtllm-serve`` - all of which speak the OpenAI HTTP protocol.

This "gateway + pluggable engine" split is how real platforms are built.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from inferlab.foundations.sampling import SamplingParams


class BackendError(RuntimeError):
    """Upstream/engine failure. Messages are safe to show to clients."""


class BackendOverloaded(BackendError):
    pass


class BackendBadRequest(BackendError):
    pass


@dataclass
class GenRequest:
    request_id: str
    params: SamplingParams
    prompt: str | list[int] | None = None  # text or token ids
    messages: list[dict] | None = None
    cache_salt: str | None = None
    guided_choice: list[str] | None = None


@dataclass
class GenChunk:
    text: str
    finished: bool = False
    finish_reason: str | None = None
    token_id: int | None = None
    logprob: float | None = None
    top_logprobs: list[tuple[str, float]] = field(default_factory=list)
    token_str: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metrics: dict = field(default_factory=dict)


class Backend(ABC):
    name: str = "base"
    model_name: str = "unknown"

    @abstractmethod
    def stream(self, req: GenRequest) -> AsyncIterator[GenChunk]: ...

    async def health(self) -> dict:
        return {"ok": True}

    def info(self) -> dict:
        return {"backend": self.name, "model": self.model_name}

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None
