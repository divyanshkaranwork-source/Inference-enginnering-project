"""Proxy to a real inference engine that speaks the OpenAI HTTP API.

All four big engines expose ``/v1/completions`` and ``/v1/chat/completions``:

==============  =========================================================  =======
engine          launch (Llama-2-7B-chat)                                    port
==============  =========================================================  =======
llama.cpp       llama-server -m llama-2-7b-chat.Q4_K_M.gguf -c 4096 -np 4   8080
vLLM            vllm serve meta-llama/Llama-2-7b-chat-hf                    8000
SGLang          python -m sglang.launch_server --model-path <same> ...      30000
TensorRT-LLM    trtllm-serve meta-llama/Llama-2-7b-chat-hf                  8000
==============  =========================================================  =======

See ``deploy/engines/`` for docker-compose files and ``docs/06-engines.md``.

Security: the upstream URL comes only from server configuration (never from
a request) so the gateway cannot be abused for SSRF; redirects are not
followed; every read has a timeout; SSE lines are size-capped; and the
request body is rebuilt from *validated* fields only (no parameter smuggling).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import httpx

from inferlab.runtime.backends.base import (
    Backend,
    BackendBadRequest,
    BackendError,
    BackendOverloaded,
    GenChunk,
    GenRequest,
)

MAX_SSE_LINE = 1 << 20  # 1 MiB

ENGINE_PRESETS: dict[str, dict] = {
    "llamacpp": {
        "title": "llama.cpp (llama-server)",
        "default_url": "http://127.0.0.1:8080",
        "health_path": "/health",
        "launch": "llama-server -m models/llama-2-7b-chat.Q4_K_M.gguf -c 4096 -np 4 --host 127.0.0.1 --port 8080",
        "extra_params": {"top_k": "top_k", "min_p": "min_p", "repetition_penalty": "repeat_penalty"},
        "strengths": "CPU/Apple Silicon/consumer GPUs, GGUF quantization (Q4_K_M...), tiny footprint",
    },
    "vllm": {
        "title": "vLLM",
        "default_url": "http://127.0.0.1:8000",
        "health_path": "/health",
        "launch": "vllm serve meta-llama/Llama-2-7b-chat-hf --max-model-len 4096 --enable-prefix-caching",
        "extra_params": {"top_k": "top_k", "min_p": "min_p", "repetition_penalty": "repetition_penalty"},
        "strengths": "PagedAttention, continuous batching, prefix caching, broad model + hardware support",
    },
    "sglang": {
        "title": "SGLang",
        "default_url": "http://127.0.0.1:30000",
        "health_path": "/health",
        "launch": "python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000",
        "extra_params": {"top_k": "top_k", "min_p": "min_p", "repetition_penalty": "repetition_penalty"},
        "strengths": "RadixAttention prefix reuse, fast structured outputs, cache-aware scheduling",
    },
    "tensorrt_llm": {
        "title": "TensorRT-LLM (trtllm-serve)",
        "default_url": "http://127.0.0.1:8000",
        "health_path": "/health",
        "launch": "trtllm-serve meta-llama/Llama-2-7b-chat-hf --max_batch_size 64 --port 8000",
        "extra_params": {"top_k": "top_k", "repetition_penalty": "repetition_penalty"},
        "strengths": "NVIDIA-optimised kernels, FP8/FP4, in-flight batching, compiled engines",
    },
}


def validate_upstream_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("upstream URL must be http(s)://host[:port]")
    if parts.username or parts.password:
        raise ValueError("do not embed credentials in the upstream URL; use INFERLAB_UPSTREAM_API_KEY")
    if parts.query or parts.fragment:
        raise ValueError("upstream URL must not contain a query or fragment")
    return url.rstrip("/")


class OpenAICompatBackend(Backend):
    def __init__(
        self,
        engine: str,
        base_url: str | None = None,
        model: str = "default",
        api_key: str | None = None,
        timeout_s: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if engine not in ENGINE_PRESETS:
            raise ValueError(f"unknown engine {engine!r}; choose from {sorted(ENGINE_PRESETS)}")
        self.name = engine
        self.preset = ENGINE_PRESETS[engine]
        self.base_url = validate_upstream_url(base_url or self.preset["default_url"])
        self.model_name = model
        headers = {"Accept": "text/event-stream"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            follow_redirects=False,
            transport=transport,
            limits=httpx.Limits(max_connections=128, max_keepalive_connections=32),
        )

    async def close(self) -> None:
        await self.client.aclose()

    def info(self) -> dict:
        return {
            "backend": self.name,
            "model": self.model_name,
            "upstream": self.base_url,
            "title": self.preset["title"],
            "launch": self.preset["launch"],
        }

    async def health(self) -> dict:
        try:
            r = await self.client.get(self.preset["health_path"], timeout=3.0)
            return {"ok": r.status_code == 200, "upstream_status": r.status_code}
        except httpx.HTTPError:
            return {"ok": False, "upstream_status": None}

    def _body(self, req: GenRequest) -> tuple[str, dict]:
        p = req.params
        body: dict = {
            "model": self.model_name,
            "max_tokens": p.max_tokens,
            "temperature": p.temperature,
            "top_p": p.top_p,
            "presence_penalty": p.presence_penalty,
            "frequency_penalty": p.frequency_penalty,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if p.stop:
            body["stop"] = list(p.stop)
        if p.seed is not None:
            body["seed"] = p.seed
        extra = self.preset["extra_params"]
        if p.top_k and "top_k" in extra:
            body[extra["top_k"]] = p.top_k
        if p.min_p and "min_p" in extra:
            body[extra["min_p"]] = p.min_p
        if p.repetition_penalty != 1.0 and "repetition_penalty" in extra:
            body[extra["repetition_penalty"]] = p.repetition_penalty
        if req.messages is not None:
            body["messages"] = [{"role": m["role"], "content": m["content"]} for m in req.messages]
            return "/v1/chat/completions", body
        body["prompt"] = req.prompt or ""
        if p.logprobs:
            body["logprobs"] = p.logprobs
        return "/v1/completions", body

    async def stream(self, req: GenRequest) -> AsyncIterator[GenChunk]:
        path, body = self._body(req)
        try:
            async with self.client.stream("POST", path, json=body) as resp:
                if resp.status_code == 429 or resp.status_code == 503:
                    raise BackendOverloaded("upstream engine is at capacity")
                if 400 <= resp.status_code < 500:
                    raise BackendBadRequest(f"upstream rejected the request ({resp.status_code})")
                if resp.status_code != 200:
                    raise BackendError(f"upstream error ({resp.status_code})")
                completion_tokens = 0
                async for data in _sse_events(resp):
                    if data == "[DONE]":
                        break
                    try:
                        ev = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(ev, dict):
                        continue
                    usage = ev.get("usage") or {}
                    for ch in ev.get("choices") or []:
                        if not isinstance(ch, dict):
                            continue
                        text = ch.get("text")
                        if text is None:
                            text = (ch.get("delta") or {}).get("content") or ""
                        if not isinstance(text, str):
                            text = ""
                        completion_tokens += 1 if text else 0
                        reason = ch.get("finish_reason")
                        yield GenChunk(
                            text=text,
                            finished=False,
                            finish_reason=reason if isinstance(reason, str) else None,
                            completion_tokens=completion_tokens,
                        )
                    if usage:
                        yield GenChunk(
                            text="",
                            finished=False,
                            prompt_tokens=int(usage.get("prompt_tokens") or 0),
                            completion_tokens=int(usage.get("completion_tokens") or completion_tokens),
                        )
                yield GenChunk(text="", finished=True)
        except httpx.TimeoutException as exc:
            raise BackendError("upstream timed out") from exc
        except httpx.HTTPError as exc:
            raise BackendError("upstream unavailable") from exc


async def _sse_events(resp: httpx.Response) -> AsyncIterator[str]:
    """Yield the ``data:`` payloads of a Server-Sent-Events stream (size-capped)."""
    buf = b""
    async for chunk in resp.aiter_bytes():
        buf += chunk
        if len(buf) > MAX_SSE_LINE and b"\n" not in buf:
            raise BackendError("upstream sent an oversized event")
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if line.startswith(b"data:"):
                yield line[5:].strip().decode("utf-8", errors="replace")
