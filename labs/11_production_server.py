"""Lab 11 - The production gateway: OpenAI API, streaming, auth, rate limits, metrics.

Runs the real FastAPI app in-process (no network needed).
Run:  python labs/11_production_server.py
Read: docs/04-production.md
"""

import json
import secrets
import warnings

from _common import explain, header

warnings.filterwarnings("ignore", category=DeprecationWarning)
from fastapi.testclient import TestClient  # noqa: E402

from inferlab.config import TINY_RANDOM  # noqa: E402
from inferlab.foundations.tokenizer import ByteTokenizer  # noqa: E402
from inferlab.foundations.weights import random_model  # noqa: E402
from inferlab.server.app import create_app  # noqa: E402
from inferlab.server.settings import Settings  # noqa: E402

key = "il-" + secrets.token_urlsafe(24)
settings = Settings(api_keys=[key], rate_limit_rps=1.0, rate_limit_burst=8)
app = create_app(settings, model=random_model(TINY_RANDOM), tokenizer=ByteTokenizer())
H = {"Authorization": f"Bearer {key}"}

with TestClient(app) as c:
    header("1. Authentication")
    print(f"  no key      -> {c.get('/v1/models').status_code}")
    print(f"  wrong key   -> {c.get('/v1/models', headers={'Authorization': 'Bearer nope'}).status_code}")
    print(f"  valid key   -> {c.get('/v1/models', headers=H).status_code}")

    header("2. OpenAI-compatible completion")
    r = c.post("/v1/completions", headers=H, json={"prompt": "Hello", "max_tokens": 8, "temperature": 0})
    print("  " + json.dumps(r.json())[:200] + " ...")

    header("3. Streaming (Server-Sent Events)")
    with c.stream("POST", "/v1/completions", headers=H,
                  json={"prompt": "Hi", "max_tokens": 4, "stream": True}) as s:
        for line in list(s.iter_lines())[:3]:
            print("  " + line[:110])
    explain("Each token is flushed as a `data: {...}` event the moment the engine samples it; the client sees "
            "TTFT ~ prefill time instead of waiting for the full answer.")

    header("4. Input validation and limits")
    print(f"  temperature=99   -> {c.post('/v1/completions', headers=H, json={'prompt': 'x', 'temperature': 99}).status_code}")
    print(f"  max_tokens=99999 -> {c.post('/v1/completions', headers=H, json={'prompt': 'x', 'max_tokens': 99999}).status_code}")
    print(f"  5 MB body        -> {c.post('/v1/completions', headers=H, content=b'x' * 5_000_000).status_code}")

    header("5. Rate limiting (token bucket: burst 8, 1 req/s)")
    codes = [c.get("/v1/models", headers=H).status_code for _ in range(6)]
    print(f"  6 rapid requests -> {codes}")

    header("6. Metrics")
    text = c.get("/metrics", headers=H).text
    for line in text.splitlines():
        if line.startswith(("inferlab_ttft_seconds_count", "inferlab_generation_tokens_total",
                            "inferlab_rate_limited_total", "inferlab_kv_cache_usage_ratio")):
            print("  " + line)
    explain("Prometheus scrapes these; alert on p99 TTFT/TPOT and queue depth, and autoscale on queue depth or KV "
            "usage rather than CPU.")
