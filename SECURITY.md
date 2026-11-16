# Security

InferLab is a teaching project, but it is built the way a production inference service should be.
This document is both the security policy and a lesson in the threat model of LLM serving.

## Reporting a vulnerability

Please report security issues privately through the repository's GitHub **Security → Report a
vulnerability** page. Do not open a public issue. Include steps to reproduce and the affected
version or commit.

## Threat model

| Asset | Threats | Controls |
|-------|---------|----------|
| GPU/CPU capacity | abuse, denial of service, cost blow-up | API keys, per-key token-bucket rate limit, request/body/token limits, bounded queues (HTTP 503 backpressure), per-request timeout, lab experiments serialised and size-capped |
| Other users' prompts | prompt leakage via the shared prefix cache (timing side channel, hash collisions) | SHA-256 block hashes (not Python `hash()`, cf. vLLM CVE-2025-25183), optional per-tenant `cache_salt`, no prompt logging |
| The host | code execution from model files or native code | no pickle/`torch.load` (raw arrays and strict safetensors only), header and size validation, path-traversal checks; C++ kernels validate pointers, sizes, overflow and block-table bounds and are tested under ASan/UBSan; ctypes library loaded only from the package directory |
| API keys | theft, brute force, timing attacks | keys stored as SHA-256 digests, constant-time comparison over all keys, ≥ 16-character keys, keys never in URLs or logs, generated keys written to a 0600 file instead of logs, UI keeps the key in `sessionStorage` only |
| Browser users | XSS, clickjacking, data exfiltration | strict CSP (`script-src 'self'`, no inline code), all UI text rendered with `textContent` (a test forbids `innerHTML`/`eval`), `frame-ancestors 'none'`, `X-Frame-Options: DENY`, `nosniff`, `Referrer-Policy: no-referrer`, CORS off by default and wildcard CORS refused |
| Upstream engines | SSRF, response smuggling | upstream URL only from operator config (http/https, no credentials, query or fragment), redirects not followed, timeouts, SSE line-size cap, request body rebuilt from validated fields only |
| Supply chain | vulnerable or tampered dependencies, models, images | pinned `requirements.txt` audited by `pip-audit`, GitHub Actions pinned by commit SHA with `permissions: contents: read`, HTTPS-only model downloads with host allow-list, size checks, pinned SHA-256 where known plus trust-on-first-use checksums, multi-stage Docker image without compilers |
| Runtime environment | privilege escalation, lateral movement | non-root UID 10001, read-only root filesystem, all capabilities dropped, `no-new-privileges`, seccomp RuntimeDefault, Kubernetes NetworkPolicy, no service-account token |

## Secure defaults (what you get without configuring anything)

- Binds to `127.0.0.1`. Binding to another address with auth disabled is refused at startup.
- Authentication is required. If no key is configured, a random key is generated and saved to
  `.inferlab_api_key` with mode 0600. It is echoed only on an interactive terminal.
- `/metrics`, `/lab/*` and `/v1/*` all require the key. Only `/`, `/static/*`, `/healthz` and
  `/readyz` are public, and they reveal nothing sensitive.
- Swagger/OpenAPI docs are disabled (`INFERLAB_ENABLE_DOCS=true` enables them).
- Error responses never include stack traces or echo the request body. Validation errors report only
  the field location and the reason.
- The access log records request id, method, path, status and latency. It never records query
  strings, headers, bodies or prompts.
- uvicorn's `Server` header is disabled.

## Running in production: your checklist

1. Put a TLS-terminating reverse proxy or ingress in front, and set `INFERLAB_HSTS=true`.
2. Provide keys through `INFERLAB_API_KEYS_FILE` from a secret store. Rotate them by updating the
   file and restarting.
3. Behind a proxy, run uvicorn with `--proxy-headers --forwarded-allow-ips=<proxy IP>` so rate
   limiting sees real client IPs. Never trust `X-Forwarded-For` from arbitrary peers.
4. Pin container images by digest (`image@sha256:...`), including the engine images in
   `deploy/engines/`.
5. Use `cache_salt` per tenant when several customers share one deployment.
6. Scrape `/metrics` with a dedicated key and alert on `inferlab_rate_limited_total` and
   `inferlab_auth_failures_total`.
7. Run `make check` (lint, tests, bandit, pip-audit, ASan/UBSan) on every change. CI does this
   automatically.

## Verification status

The latest run of `make check` reported: ruff clean (including the `S` security rules), bandit with
**0 issues and 0 suppressions**, pip-audit with **no known vulnerabilities** in the runtime and dev
locks, C++ built with `-Wall -Wextra -Wconversion` and hardening flags (`-fstack-protector-strong`,
`_FORTIFY_SOURCE=2`, full RELRO, non-executable stack) with zero warnings, C++ tests clean under
AddressSanitizer + UndefinedBehaviorSanitizer, and all 136 Python tests passing. Those tests include
the auth, rate-limit, body-limit, CSP, input-validation, path-traversal, malformed-weight-file and
out-of-bounds-kernel cases.

## Known limitations

- The rate limiter and API-key store are in-memory and per process. With several replicas, each
  enforces its own limits; use a shared store such as Redis, or an API gateway, for global quotas.
- `stories*.bin` checkpoints have no published SHA-256. They are verified by exact size and strict
  format parsing, and their hash is recorded on first download (trust on first use). The tokenizer is
  pinned to a commit and its SHA-256.
- The CUDA/Triton kernels in `gpu_kernels/` are educational, are not built in CI, and do not
  bounds-check block tables the way the C++ kernels do.
