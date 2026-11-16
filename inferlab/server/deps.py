"""FastAPI dependencies: authentication + rate limiting on every protected route."""

from __future__ import annotations

from fastapi import HTTPException, Request


def _client_ip(request: Request) -> str:
    # request.client is the TCP peer. Behind a reverse proxy, run uvicorn with
    # --proxy-headers --forwarded-allow-ips=<proxy ip> so this is the real
    # client - never trust X-Forwarded-For from arbitrary peers.
    return request.client.host if request.client else "unknown"


def authenticate(request: Request) -> str:
    state = request.app.state
    principal = state.auth.verify(state.auth.extract(request.headers))
    if principal is None:
        state.metrics.auth_failures.inc()
        raise HTTPException(401, "missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})
    if principal == "anonymous":
        principal = "ip-" + _client_ip(request)
    return principal


def rate_limit(request: Request, principal: str, cost: float = 1.0) -> None:
    state = request.app.state
    ok, retry = state.limiter.allow(principal, cost)
    if not ok:
        state.metrics.rate_limited.inc()
        raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(max(1, int(retry + 0.999)))})


async def require_auth_and_rate_limit(request: Request) -> str:
    principal = authenticate(request)
    rate_limit(request, principal)
    return principal


async def require_auth_heavy(request: Request) -> str:
    """Lab endpoints that burn CPU cost 5 rate-limit tokens."""
    principal = authenticate(request)
    rate_limit(request, principal, cost=5.0)
    return principal
