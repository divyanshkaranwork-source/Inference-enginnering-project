"""The built-in engine as a Backend."""

from __future__ import annotations

from collections.abc import AsyncIterator

from inferlab.foundations.tokenizer import BaseTokenizer
from inferlab.runtime.async_engine import AsyncEngine, EngineOverloaded
from inferlab.runtime.backends.base import (
    Backend,
    BackendBadRequest,
    BackendError,
    BackendOverloaded,
    GenChunk,
    GenRequest,
)
from inferlab.runtime.chat import apply_chat_template
from inferlab.runtime.engine import LLMEngine


class NativeBackend(Backend):
    name = "native"

    def __init__(self, engine: LLMEngine, max_waiting: int = 64):
        self.engine = engine
        self.tok: BaseTokenizer = engine.tok
        self.model_name = engine.model.cfg.name
        self.async_engine = AsyncEngine(engine, max_waiting=max_waiting)

    async def start(self) -> None:
        self.async_engine.start()

    async def close(self) -> None:
        self.async_engine.shutdown()

    async def health(self) -> dict:
        return {"ok": self.async_engine.is_alive, "engine_errors": self.async_engine.errors}

    def info(self) -> dict:
        cfg = self.engine.model.cfg
        return {
            "backend": self.name,
            "model": self.model_name,
            "architecture": cfg.to_dict(),
            "weights_mb": self.engine.model.nbytes / 2**20,
            "max_model_len": self.engine.max_model_len,
            "kernels": "cpp" if self.engine.model.kernels is not None else "numpy",
        }

    async def stream(self, req: GenRequest) -> AsyncIterator[GenChunk]:
        try:
            if req.messages is not None:
                prompt = apply_chat_template(self.tok, req.messages, self.engine.model.cfg.chat_template)
            else:
                prompt = req.prompt or ""
            gen = self.async_engine.generate(
                prompt, req.params, req.request_id, req.cache_salt, req.guided_choice
            )
            async for out in gen:
                yield GenChunk(
                    text=out.text,
                    finished=out.finished,
                    finish_reason=out.finish_reason,
                    token_id=out.token_id,
                    logprob=out.logprob,
                    token_str=self.tok.id_to_piece(out.token_id) if out.token_id is not None else None,
                    top_logprobs=[(self.tok.id_to_piece(t), lp) for t, lp in out.top_logprobs],
                    prompt_tokens=out.prompt_tokens,
                    completion_tokens=out.completion_tokens,
                    metrics=out.metrics,
                )
        except EngineOverloaded as exc:
            raise BackendOverloaded("server is at capacity, retry later") from exc
        except ValueError as exc:
            raise BackendBadRequest(str(exc)) from exc
        except RuntimeError as exc:
            raise BackendError("engine failure") from exc
