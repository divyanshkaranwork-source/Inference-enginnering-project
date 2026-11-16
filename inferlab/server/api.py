"""OpenAI-compatible endpoints: /v1/models, /v1/completions, /v1/chat/completions.

Works with the official ``openai`` Python client, LangChain, curl, etc.:

    curl -N http://127.0.0.1:8000/v1/completions \\
      -H "Authorization: Bearer $INFERLAB_API_KEY" -H "Content-Type: application/json" \\
      -d '{"prompt": "Once upon a time", "max_tokens": 64, "stream": true}'
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import aclosing

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from inferlab.runtime.backends import (
    BackendBadRequest,
    BackendError,
    BackendOverloaded,
    GenChunk,
    GenRequest,
)
from inferlab.server.deps import require_auth_and_rate_limit
from inferlab.server.schemas import ChatRequest, CompletionRequest

router = APIRouter()


class _Timeout(Exception):
    pass


async def _generate(request: Request, req: GenRequest):
    """Wrap a backend stream with a deadline, metrics and error mapping."""
    app = request.app.state
    backend, metrics, settings = app.backend, app.metrics, app.settings
    t0 = time.monotonic()
    deadline = t0 + settings.request_timeout_s
    ttft = None
    n_tokens = 0
    prompt_tokens = 0
    metrics.inflight.inc()
    try:
        async with aclosing(backend.stream(req)) as it:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _Timeout()
                try:
                    chunk: GenChunk = await asyncio.wait_for(anext(it), remaining)
                except StopAsyncIteration:
                    break
                except TimeoutError as exc:
                    raise _Timeout() from exc
                if chunk.text or chunk.token_id is not None:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    n_tokens = max(n_tokens + 1, chunk.completion_tokens)
                prompt_tokens = max(prompt_tokens, chunk.prompt_tokens)
                yield chunk
                if chunk.finished:
                    break
        metrics.observe_request(ttft, time.monotonic() - t0, n_tokens, prompt_tokens)
    finally:
        metrics.inflight.dec()


def _error(status: int, message: str, etype: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": etype}}, status_code=status, headers=headers)


def _map_exc(exc: Exception) -> tuple[int, str, str]:
    if isinstance(exc, BackendOverloaded):
        return 503, str(exc), "overloaded"
    if isinstance(exc, BackendBadRequest):
        return 400, str(exc), "invalid_request_error"
    if isinstance(exc, _Timeout):
        return 504, "request timed out", "timeout"
    if isinstance(exc, BackendError):
        return 502, str(exc), "backend_error"
    return 500, "internal error", "internal_error"


def _limits(request: Request, max_tokens: int | None, prompt_chars: int) -> int:
    s = request.app.state.settings
    if prompt_chars > s.max_prompt_chars:
        raise HTTPException(400, f"prompt too long: {prompt_chars} > {s.max_prompt_chars} characters")
    mt = max_tokens or 16
    if mt > s.max_tokens_limit:
        raise HTTPException(400, f"max_tokens must be <= {s.max_tokens_limit}")
    return mt


def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n".encode()


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


# ---------------------------------------------------------------------------
@router.get("/v1/models")
async def list_models(request: Request, _principal: str = Depends(require_auth_and_rate_limit)):
    b = request.app.state.backend
    return {"object": "list", "data": [{"id": b.model_name, "object": "model", "owned_by": "inferlab",
                                        "created": int(request.app.state.started_at)}]}


@router.post("/v1/completions")
async def completions(body: CompletionRequest, request: Request,
                      principal: str = Depends(require_auth_and_rate_limit)):
    prompt_chars = len(body.prompt) if isinstance(body.prompt, str) else len(body.prompt) * 4
    max_tokens = _limits(request, body.max_tokens, prompt_chars)
    rid = f"cmpl-{uuid.uuid4().hex}"
    req = GenRequest(
        request_id=rid,
        params=body.sampling(max_tokens, body.logprobs or 0),
        prompt=body.prompt,
        cache_salt=body.cache_salt,
        guided_choice=body.guided_choice,
    )
    model = request.app.state.backend.model_name
    created = int(time.time())

    def logprobs_obj(c: GenChunk):
        if not body.logprobs or c.token_str is None:
            return None
        return {"tokens": [c.token_str], "token_logprobs": [c.logprob],
                "top_logprobs": [dict(c.top_logprobs)]}

    if body.stream:
        async def stream():
            finish = None
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            try:
                async for c in _generate(request, req):
                    finish = c.finish_reason or finish
                    usage = {"prompt_tokens": max(usage["prompt_tokens"], c.prompt_tokens),
                             "completion_tokens": max(usage["completion_tokens"], c.completion_tokens)}
                    if not c.text and c.token_id is None and not c.finished:
                        continue
                    choice = {"index": 0, "text": c.text, "logprobs": logprobs_obj(c),
                              "finish_reason": c.finish_reason if c.finished else None}
                    ev = {"id": rid, "object": "text_completion", "created": created, "model": model,
                          "choices": [choice]}
                    if c.metrics:
                        ev["x_metrics"] = c.metrics
                    yield _sse(ev)
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                yield _sse({"id": rid, "object": "text_completion", "created": created, "model": model,
                            "choices": [{"index": 0, "text": "", "logprobs": None,
                                         "finish_reason": finish or "stop"}], "usage": usage})
            except Exception as exc:  # report errors in-band once streaming has started
                status, msg, etype = _map_exc(exc)
                yield _sse({"error": {"message": msg, "type": etype, "code": status}})
            yield b"data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)

    text, finish, lp_tokens, lp_vals, lp_top = [], None, [], [], []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    try:
        async for c in _generate(request, req):
            text.append(c.text)
            finish = c.finish_reason or finish
            usage = {"prompt_tokens": max(usage["prompt_tokens"], c.prompt_tokens),
                     "completion_tokens": max(usage["completion_tokens"], c.completion_tokens)}
            if body.logprobs and c.token_str is not None:
                lp_tokens.append(c.token_str)
                lp_vals.append(c.logprob)
                lp_top.append(dict(c.top_logprobs))
    except Exception as exc:
        status, msg, etype = _map_exc(exc)
        return _error(status, msg, etype, {"Retry-After": "1"} if status == 503 else None)
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    lp = {"tokens": lp_tokens, "token_logprobs": lp_vals, "top_logprobs": lp_top} if body.logprobs else None
    return {"id": rid, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": "".join(text), "logprobs": lp, "finish_reason": finish or "stop"}],
            "usage": usage}


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatRequest, request: Request,
                           principal: str = Depends(require_auth_and_rate_limit)):
    prompt_chars = sum(len(m.content) for m in body.messages)
    max_tokens = _limits(request, body.max_completion_tokens or body.max_tokens, prompt_chars)
    rid = f"chatcmpl-{uuid.uuid4().hex}"
    n_lp = (body.top_logprobs or 0) if body.logprobs else 0
    req = GenRequest(
        request_id=rid,
        params=body.sampling(max_tokens, n_lp),
        messages=[{"role": m.role, "content": m.content} for m in body.messages],
        cache_salt=body.cache_salt,
        guided_choice=body.guided_choice,
    )
    model = request.app.state.backend.model_name
    created = int(time.time())

    if body.stream:
        async def stream():
            finish = None
            first = True
            try:
                async for c in _generate(request, req):
                    finish = c.finish_reason or finish
                    if not c.text and not c.finished and not first:
                        continue
                    delta = {"content": c.text}
                    if first:
                        delta["role"] = "assistant"
                        first = False
                    ev = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                          "choices": [{"index": 0, "delta": delta,
                                       "finish_reason": c.finish_reason if c.finished else None}]}
                    if c.metrics:
                        ev["x_metrics"] = c.metrics
                    yield _sse(ev)
                yield _sse({"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish or "stop"}]})
            except Exception as exc:
                status, msg, etype = _map_exc(exc)
                yield _sse({"error": {"message": msg, "type": etype, "code": status}})
            yield b"data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)

    text, finish = [], None
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    try:
        async for c in _generate(request, req):
            text.append(c.text)
            finish = c.finish_reason or finish
            usage = {"prompt_tokens": max(usage["prompt_tokens"], c.prompt_tokens),
                     "completion_tokens": max(usage["completion_tokens"], c.completion_tokens)}
    except Exception as exc:
        status, msg, etype = _map_exc(exc)
        return _error(status, msg, etype, {"Retry-After": "1"} if status == 503 else None)
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return {"id": rid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(text)},
                         "finish_reason": finish or "stop"}],
            "usage": usage}
