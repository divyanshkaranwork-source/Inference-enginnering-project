import json
import os
import re
import stat
import warnings

import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)
from fastapi.testclient import TestClient  # noqa: E402

from inferlab.server.app import create_app  # noqa: E402
from inferlab.server.security import APIKeyAuth, RateLimiter  # noqa: E402
from inferlab.server.settings import Settings  # noqa: E402

KEY = "test-key-0123456789abcdef"
H = {"Authorization": f"Bearer {KEY}"}


def make_client(tiny_model, byte_tok, **kw):
    s = Settings(api_keys=[KEY], rate_limit_rps=1000, rate_limit_burst=1000, **kw)
    return TestClient(create_app(s, model=tiny_model, tokenizer=byte_tok))


@pytest.fixture(scope="module")
def client(tiny_model, byte_tok):
    with make_client(tiny_model, byte_tok) as c:
        yield c


# ---------------------------------------------------------------------------
# authentication & hardening
# ---------------------------------------------------------------------------
def test_health_is_public_but_api_is_not(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 200
    r = client.get("/v1/models")
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert client.get("/v1/models", headers={"Authorization": "Bearer wrong-key-000000000"}).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": f"Basic {KEY}"}).status_code == 401
    assert client.get("/v1/models", headers={"X-API-Key": KEY}).status_code == 200
    assert client.get("/v1/models", headers=H).json()["data"][0]["id"] == "tiny-random"


def test_lab_and_metrics_require_auth(client):
    assert client.get("/lab/overview").status_code == 401
    assert client.post("/lab/tokenize", json={"text": "x"}).status_code == 401
    assert client.get("/metrics").status_code == 401
    assert "inferlab_requests" in client.get("/metrics", headers=H).text


def test_security_headers_and_ui(client):
    r = client.get("/")
    assert r.status_code == 200 and "<title>" in r.text
    csp = r.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp and "frame-ancestors 'none'" in csp
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert "x-request-id" in r.headers
    assert "server" not in {k.lower() for k in r.headers if k.lower() == "server" and "uvicorn" in r.headers[k]}


def test_docs_disabled_by_default(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_no_inline_script_in_ui():
    from inferlab.server.app import STATIC_DIR

    html = (STATIC_DIR / "index.html").read_text()
    assert "<script>" not in html and "onclick" not in html and "style=" not in html
    for js in (STATIC_DIR / "js").rglob("*.js"):
        src = js.read_text()
        for sink in (r"\.innerHTML\b", r"\.outerHTML\b", r"insertAdjacentHTML", r"document\.write",
                     r"\beval\(", r"new Function"):
            assert not re.search(sink, src), f"{js.name}: {sink}"


def test_body_size_limit(tiny_model, byte_tok):
    with make_client(tiny_model, byte_tok, max_body_bytes=2048) as c:
        r = c.post("/v1/completions", headers=H, content=b"{" + b" " * 5000 + b"}")
        assert r.status_code == 413


def test_validation_errors_do_not_echo_input(client):
    secret = "SECRET-PROMPT-CONTENT"
    r = client.post("/v1/completions", headers=H, json={"prompt": secret, "temperature": 99})
    assert r.status_code == 422 and secret not in r.text
    assert r.json()["error"]["details"][0]["loc"] == ["body", "temperature"]


def test_request_limits(client):
    assert client.post("/v1/completions", headers=H, json={"prompt": "x" * 40000}).status_code == 400
    assert client.post("/v1/completions", headers=H, json={"prompt": "x", "max_tokens": 5000}).status_code == 400
    assert client.post("/v1/completions", headers=H, json={"prompt": "x", "n": 2}).status_code == 422
    assert client.post("/v1/completions", headers=H, json={"prompt": "x", "stop": ["a"] * 5}).status_code == 422
    assert client.post("/v1/completions", headers=H, json={"prompt": ""}).status_code == 422


def test_rate_limit(tiny_model, byte_tok):
    s = Settings(api_keys=[KEY], rate_limit_rps=0.01, rate_limit_burst=2)
    with TestClient(create_app(s, model=tiny_model, tokenizer=byte_tok)) as c:
        codes = [c.get("/v1/models", headers=H).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        r = c.get("/v1/models", headers=H)
        assert int(r.headers["retry-after"]) >= 1


# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------
def test_completion(client):
    r = client.post("/v1/completions", headers=H,
                    json={"prompt": "Hello", "max_tokens": 6, "temperature": 0, "logprobs": 2})
    j = r.json()
    assert r.status_code == 200 and j["object"] == "text_completion"
    assert j["usage"]["completion_tokens"] == 6 and j["choices"][0]["finish_reason"] == "length"
    assert len(j["choices"][0]["logprobs"]["tokens"]) == 6


def test_completion_with_token_ids(client):
    r = client.post("/v1/completions", headers=H, json={"prompt": [1, 75, 104], "max_tokens": 3, "temperature": 0})
    assert r.status_code == 200 and r.json()["usage"]["prompt_tokens"] == 3
    bad = client.post("/v1/completions", headers=H, json={"prompt": [1, 99999], "max_tokens": 3})
    assert bad.status_code == 400


def test_streaming_completion(client):
    with client.stream("POST", "/v1/completions", headers=H,
                       json={"prompt": "Hi", "max_tokens": 5, "stream": True, "temperature": 0}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(c["choices"][0]["text"] for c in chunks)
    nonstream = client.post("/v1/completions", headers=H, json={"prompt": "Hi", "max_tokens": 5, "temperature": 0})
    assert text == nonstream.json()["choices"][0]["text"]
    assert chunks[-1]["usage"]["completion_tokens"] == 5


def test_chat_completion_and_stream(client):
    body = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "Hi"}], "max_tokens": 4}
    j = client.post("/v1/chat/completions", headers=H, json=body).json()
    assert j["object"] == "chat.completion" and j["choices"][0]["message"]["role"] == "assistant"
    with client.stream("POST", "/v1/chat/completions", headers=H, json={**body, "stream": True}) as r:
        events = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    first = json.loads(events[0])
    assert first["object"] == "chat.completion.chunk" and first["choices"][0]["delta"]["role"] == "assistant"
    bad = client.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "root", "content": "x"}]})
    assert bad.status_code == 422


def test_guided_choice_via_api(client):
    j = client.post("/v1/completions", headers=H, json={"prompt": "Q:", "max_tokens": 10, "temperature": 1,
                                                        "guided_choice": ["red", "blue"]}).json()
    assert j["choices"][0]["text"] in ("red", "blue")


def test_metrics_record_requests(client):
    client.post("/v1/completions", headers=H, json={"prompt": "Hello", "max_tokens": 3})
    text = client.get("/metrics", headers=H).text
    assert "inferlab_ttft_seconds_count" in text and "inferlab_kv_cache_usage_ratio" in text
    summary = client.get("/lab/metrics/summary", headers=H).json()
    assert summary["requests"] >= 1 and summary["ttft_ms"]["p50"] > 0


# ---------------------------------------------------------------------------
# lab endpoints
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path,body", [
    ("/lab/tokenize", {"text": "hello"}),
    ("/lab/sampling", {"prompt": "hi", "top_k": 5}),
    ("/lab/memory", {"model": "llama-2-7b"}),
    ("/lab/roofline", {"gpu": "H100-SXM-80GB"}),
    ("/lab/kv-cache", {"new_tokens": 6}),
    ("/lab/quantization", {"kinds": ["fp32", "q4_0"], "new_tokens": 3}),
    ("/lab/radix", {"prompts": ["abc", "abd"]}),
    ("/lab/speculative", {"max_tokens": 8, "method": "draft_model", "draft_layers": 2}),
    ("/lab/batching-compare", {"num_requests": 3, "max_tokens": 6}),
    ("/lab/kernels/bench", {"op": "softmax"}),
    ("/lab/parallel/plan", {"tp": 2}),
    ("/lab/parallel/recommend", {"n_gpus": 4}),
    ("/lab/parallel/simulate", {"strategy": "pp", "degree": 2}),
    ("/lab/parallel/simulate", {"strategy": "ep", "degree": 4, "capacity_factor": 1.0}),
    ("/lab/engine/burst", {"n": 2, "max_tokens": 4, "stagger_ms": 0}),
])
def test_lab_endpoints(client, path, body):
    r = client.post(path, headers=H, json=body)
    assert r.status_code == 200, r.text


def test_lab_gets(client):
    for p in ("/lab/overview", "/lab/engine/state", "/lab/engines"):
        assert client.get(p, headers=H).status_code == 200


def test_lab_rejects_unknown_fields_and_bad_values(client):
    assert client.post("/lab/tokenize", headers=H, json={"text": "x", "evil": 1}).status_code == 422
    assert client.post("/lab/memory", headers=H, json={"model": "gpt-9"}).status_code == 400
    assert client.post("/lab/kernels/bench", headers=H, json={"op": "rm -rf"}).status_code == 422
    assert client.post("/lab/parallel/simulate", headers=H, json={"strategy": "tp", "degree": 3}).status_code == 400
    assert client.post("/lab/engine/burst", headers=H, json={"n": 1000}).status_code == 422


# ---------------------------------------------------------------------------
# settings & security primitives
# ---------------------------------------------------------------------------
def test_settings_refuse_insecure_configs():
    with pytest.raises(ValueError):
        Settings(host="0.0.0.0", auth_disabled=True).validate()  # nosec B104 - testing the refusal
    with pytest.raises(ValueError):
        Settings(api_keys=["short"]).validate()
    with pytest.raises(ValueError):
        Settings(api_keys=[KEY], cors_origins=["*"]).validate()
    Settings(host="127.0.0.1", auth_disabled=True).validate()


def test_generated_key_file_is_private(tmp_path, monkeypatch, tiny_model, byte_tok):
    key_file = tmp_path / "key"
    monkeypatch.setenv("INFERLAB_GENERATED_KEY_FILE", str(key_file))
    s = Settings()
    app = create_app(s, model=tiny_model, tokenizer=byte_tok)
    assert s.generated_api_key and key_file.read_text().strip() == s.generated_api_key
    assert stat.S_IMODE(os.stat(key_file).st_mode) == 0o600
    with TestClient(app) as c:
        assert c.get("/v1/models", headers={"Authorization": f"Bearer {s.generated_api_key}"}).status_code == 200


def test_api_key_auth_unit():
    a = APIKeyAuth([KEY, "another-key-0123456789"])
    assert a.verify(KEY).startswith("key-") and a.verify("nope") is None and a.verify(None) is None
    assert a.verify("x" * 10_000) is None
    assert APIKeyAuth([], disabled=True).verify(None) == "anonymous"


def test_rate_limiter_bounded_memory():
    rl = RateLimiter(rate=1, burst=1, max_principals=100)
    for i in range(1000):
        rl.allow(f"p{i}")
    assert len(rl._b) == 100
