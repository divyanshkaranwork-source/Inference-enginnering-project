"""Security building blocks for the gateway.

* ``APIKeyAuth``   - Bearer / X-API-Key authentication. Only SHA-256 digests
  of keys are kept in memory; comparison is constant-time and always checks
  every key (no early exit -> no timing oracle on which key matched).
* ``RateLimiter``  - token bucket per principal, bounded memory (LRU).
* ``SecurityHeadersMiddleware`` - CSP, nosniff, frame denial, no referrer...
* ``BodySizeLimitMiddleware``   - rejects oversized bodies *while streaming*,
  not only via Content-Length (which a client can omit or lie about).

All middlewares are raw ASGI so they also cover streaming responses.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from collections import OrderedDict

from starlette.types import ASGIApp, Message, Receive, Scope, Send

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; "
    "form-action 'self'; frame-ancestors 'none'"
)


def _digest(key: str) -> bytes:
    return hashlib.sha256(key.encode("utf-8")).digest()


def generate_api_key() -> str:
    return "il-" + secrets.token_urlsafe(32)


class APIKeyAuth:
    def __init__(self, keys: list[str], disabled: bool = False):
        self.disabled = disabled
        self._digests = [_digest(k) for k in keys]

    @property
    def enabled(self) -> bool:
        return not self.disabled

    @staticmethod
    def extract(headers) -> str | None:
        auth = headers.get("authorization")
        if auth:
            scheme, _, value = auth.partition(" ")
            if scheme.lower() == "bearer" and value.strip():
                return value.strip()
            return None
        key = headers.get("x-api-key")
        return key.strip() if key else None

    def verify(self, presented: str | None) -> str | None:
        """Return a non-secret principal id for a valid key, else None."""
        if self.disabled:
            return "anonymous"
        if not presented or len(presented) > 512:
            return None
        d = _digest(presented)
        match = None
        for ref in self._digests:  # check all keys: constant work regardless of which matches
            if hmac.compare_digest(d, ref):
                match = ref
        return None if match is None else "key-" + match.hex()[:12]


class RateLimiter:
    """Token bucket: ``rate`` tokens/s refill, ``burst`` capacity, per principal."""

    def __init__(self, rate: float, burst: int, max_principals: int = 10_000):
        self.rate, self.burst, self.max = rate, burst, max_principals
        self._b: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()
        self.rejected = 0

    def allow(self, principal: str, cost: float = 1.0) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            tokens, last = self._b.pop(principal, (float(self.burst), now))
            tokens = min(float(self.burst), tokens + (now - last) * self.rate)
            ok = tokens >= cost
            if ok:
                tokens -= cost
            self._b[principal] = (tokens, now)
            while len(self._b) > self.max:  # bounded memory under key/IP churn
                self._b.popitem(last=False)
            if not ok:
                self.rejected += 1
                return False, (cost - tokens) / self.rate
            return True, 0.0


# ---------------------------------------------------------------------------
# ASGI middlewares
# ---------------------------------------------------------------------------
class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp, hsts: bool = False):
        self.app = app
        self.headers = [
            (b"content-security-policy", CSP.encode()),
            (b"x-content-type-options", b"nosniff"),
            (b"x-frame-options", b"DENY"),
            (b"referrer-policy", b"no-referrer"),
            (b"permissions-policy", b"camera=(), microphone=(), geolocation=(), payment=()"),
            (b"cross-origin-opener-policy", b"same-origin"),
            (b"cross-origin-resource-policy", b"same-origin"),
        ]
        if hsts:
            self.headers.append((b"strict-transport-security", b"max-age=63072000; includeSubDomains"))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                existing = {k.lower() for k, _ in message.get("headers", [])}
                extra = [(k, v) for k, v in self.headers if k not in existing]
                message = {**message, "headers": list(message.get("headers", [])) + extra}
            await send(message)

        await self.app(scope, receive, send_wrapper)


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for k, v in scope.get("headers", []):
            if k == b"content-length":
                try:
                    too_big = int(v) > self.max_bytes
                except ValueError:
                    too_big = True
                if too_big:
                    await _reject(send, 413, "request body too large")
                    return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await _reject(send, 413, "request body too large")


class _BodyTooLarge(Exception):
    pass


async def _reject(send: Send, status: int, msg: str) -> None:
    body = json.dumps({"error": {"message": msg, "type": "invalid_request_error"}}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})
