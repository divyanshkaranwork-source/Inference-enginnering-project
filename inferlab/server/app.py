"""Application factory for the InferLab gateway.

    python -m inferlab serve            # or: inferlab serve
    INFERLAB_BACKEND=vllm INFERLAB_UPSTREAM_URL=http://127.0.0.1:8000 inferlab serve --port 9000

Layout:
    /                       web UI (static, strict CSP)
    /healthz  /readyz       liveness / readiness probes (no auth, no details)
    /metrics                Prometheus (auth unless INFERLAB_METRICS_PUBLIC=true)
    /v1/...                 OpenAI-compatible API
    /lab/...                interactive experiments used by the UI
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException

from inferlab import __version__
from inferlab.server.deps import authenticate
from inferlab.server.metrics import Metrics
from inferlab.server.security import (
    APIKeyAuth,
    BodySizeLimitMiddleware,
    RateLimiter,
    SecurityHeadersMiddleware,
    generate_api_key,
)
from inferlab.server.settings import Settings

logger = logging.getLogger("inferlab.server")
STATIC_DIR = Path(__file__).resolve().parent / "static"


def build_backend(settings: Settings, model, tokenizer):
    from inferlab.runtime.backends import NativeBackend, OpenAICompatBackend
    from inferlab.runtime.engine import EngineConfig, LLMEngine

    if settings.backend == "native":
        engine = LLMEngine(model, tokenizer, EngineConfig(
            block_size=settings.block_size,
            num_kv_blocks=settings.num_kv_blocks,
            max_num_seqs=settings.max_num_seqs,
            max_num_batched_tokens=settings.max_num_batched_tokens,
            enable_prefix_caching=settings.enable_prefix_caching,
            enable_chunked_prefill=settings.enable_chunked_prefill,
        ))
        return NativeBackend(engine, max_waiting=settings.max_waiting)
    return OpenAICompatBackend(settings.backend, settings.upstream_url, settings.upstream_model,
                               settings.upstream_api_key, settings.request_timeout_s)


def load_model(settings: Settings):
    from inferlab import models
    from inferlab.kernels import get_kernels

    return models.load(settings.model, settings.tokenizer, settings.linear_kind, get_kernels(settings.kernels),
                       Path(settings.models_dir))


class RequestLogMiddleware:
    """Access log without query strings, bodies or credentials; adds X-Request-ID."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        status = {"code": 0}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                message = {**message, "headers": list(message.get("headers", [])) + [(b"x-request-id", rid.encode())]}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            logger.info("%s %s %s %d %.1fms", rid, scope.get("method"), scope.get("path"), status["code"],
                        (time.perf_counter() - t0) * 1e3)


def _announce_generated_key(key: str) -> None:
    """Hand the operator the generated key without writing it into log pipelines.

    The key goes to a 0600 file; it is echoed to the terminal only when stderr
    is an interactive TTY (never in containers, where stderr is shipped to logs).
    """
    path = Path(os.environ.get("INFERLAB_GENERATED_KEY_FILE", ".inferlab_api_key"))
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(key + "\n")
        where = f"saved to {path.resolve()} (mode 600)"
    except OSError:
        where = "could not be saved to a file"
    msg = f"[inferlab] No INFERLAB_API_KEYS configured - generated a key for this run, {where}."
    if sys.stderr.isatty():
        msg += f"\n[inferlab]   {key}\n[inferlab] Paste it into the UI or send it as 'Authorization: Bearer <key>'."
    print(msg, file=sys.stderr, flush=True)


def create_app(settings: Settings | None = None, model=None, tokenizer=None, backend=None) -> FastAPI:
    settings = (settings or Settings.from_env()).validate()
    if not settings.auth_disabled and not settings.api_keys:
        settings.generated_api_key = generate_api_key()
        settings.api_keys = [settings.generated_api_key]
        _announce_generated_key(settings.generated_api_key)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal model, tokenizer, backend
        from inferlab.server.lab import LabContext

        if model is None or tokenizer is None:
            model, tokenizer = await asyncio.to_thread(load_model, settings)
        if backend is None:
            backend = build_backend(settings, model, tokenizer)
        app.state.backend = backend
        app.state.lab = LabContext(model, tokenizer, settings)
        snap = backend.async_engine.snapshot if hasattr(backend, "async_engine") else None
        app.state.metrics = Metrics(snap)
        await backend.start()
        app.state.ready = True
        logger.info("serving %s via %s backend", backend.model_name, backend.name)
        try:
            yield
        finally:
            app.state.ready = False
            await backend.close()

    docs = settings.enable_docs
    app = FastAPI(
        title="InferLab",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs else None,
    )
    app.state.settings = settings
    app.state.auth = APIKeyAuth(settings.api_keys, settings.auth_disabled)
    app.state.limiter = RateLimiter(settings.rate_limit_rps, settings.rate_limit_burst)
    app.state.metrics = Metrics()
    app.state.lab_semaphore = asyncio.Semaphore(1)
    app.state.burst_tasks = set()
    app.state.started_at = time.time()
    app.state.ready = False

    # ---- errors: never leak stack traces or echo request bodies -----------------
    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        etype = {401: "authentication_error", 429: "rate_limit_error", 404: "not_found"}.get(
            exc.status_code, "invalid_request_error")
        return JSONResponse({"error": {"message": str(exc.detail), "type": etype}}, status_code=exc.status_code,
                            headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        details = [{"loc": [str(x) for x in e.get("loc", [])], "msg": e.get("msg", "")} for e in exc.errors()][:10]
        return JSONResponse({"error": {"message": "invalid request", "type": "invalid_request_error",
                                       "details": details}}, status_code=422)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("unhandled error on %s", request.url.path)
        return JSONResponse({"error": {"message": "internal error", "type": "internal_error"}}, status_code=500)

    # ---- routes -----------------------------------------------------------------
    from inferlab.server import api, lab

    app.include_router(api.router)
    if settings.enable_lab:
        app.include_router(lab.router)

    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz(request: Request):
        if not request.app.state.ready:
            return JSONResponse({"status": "starting"}, status_code=503)
        h = await request.app.state.backend.health()
        return JSONResponse({"status": "ready" if h.get("ok") else "degraded"}, status_code=200 if h.get("ok") else 503)

    @app.get("/metrics", include_in_schema=False)
    async def metrics(request: Request):
        if not settings.metrics_public:
            authenticate(request)
        return Response(generate_latest(request.app.state.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/v1", include_in_schema=False)
    async def v1_root():
        raise HTTPException(404, "see /v1/models, /v1/completions, /v1/chat/completions")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ---- middleware (outermost last) ----------------------------------------------
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                           allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(SecurityHeadersMiddleware, hsts=settings.hsts)
    app.add_middleware(RequestLogMiddleware)
    return app
