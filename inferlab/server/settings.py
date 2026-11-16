"""Server configuration from environment variables (12-factor style).

Every setting is ``INFERLAB_<NAME>``. Secure defaults:

* binds to 127.0.0.1 (opt in to 0.0.0.0, e.g. inside a container),
* authentication ON - if no keys are configured a random key is generated
  at start-up and printed once,
* auth may only be disabled on a loopback address,
* CORS off, Swagger UI off, metrics require auth, strict request limits.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from pathlib import Path

BACKENDS = ("native", "llamacpp", "vllm", "sglang", "tensorrt_llm")


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int, lo: int, hi: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        x = int(v)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not lo <= x <= hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}]")
    return x


def _float(name: str, default: float, lo: float, hi: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        x = float(v)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not lo <= x <= hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}]")
    return x


def _list(name: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


def is_loopback(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000
    backend: str = "native"
    model: str = "auto"
    tokenizer: str | None = None
    models_dir: str = "models"
    linear_kind: str = "fp32"
    kernels: str = "numpy"
    # engine
    num_kv_blocks: int = 256
    block_size: int = 16
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 256
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    max_waiting: int = 64
    # upstream engines
    upstream_url: str | None = None
    upstream_model: str = "default"
    upstream_api_key: str | None = field(default=None, repr=False)
    # security
    api_keys: list[str] = field(default_factory=list, repr=False)
    auth_disabled: bool = False
    rate_limit_rps: float = 5.0
    rate_limit_burst: int = 20
    max_body_bytes: int = 1_048_576
    max_prompt_chars: int = 32_000
    max_tokens_limit: int = 1024
    request_timeout_s: float = 120.0
    cors_origins: list[str] = field(default_factory=list)
    enable_docs: bool = False
    enable_lab: bool = True
    metrics_public: bool = False
    hsts: bool = False
    generated_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> Settings:
        keys = _list("INFERLAB_API_KEYS")
        key_file = os.environ.get("INFERLAB_API_KEYS_FILE")
        if key_file:  # e.g. a Docker/Kubernetes secret mounted as a file
            keys += [k.strip() for k in Path(key_file).read_text("utf-8").splitlines() if k.strip()]
        s = cls(
            host=os.environ.get("INFERLAB_HOST", "127.0.0.1"),
            port=_int("INFERLAB_PORT", 8000, 1, 65535),
            backend=os.environ.get("INFERLAB_BACKEND", "native"),
            model=os.environ.get("INFERLAB_MODEL", "auto"),
            tokenizer=os.environ.get("INFERLAB_TOKENIZER") or None,
            models_dir=os.environ.get("INFERLAB_MODELS_DIR", "models"),
            linear_kind=os.environ.get("INFERLAB_WEIGHT_DTYPE", "fp32"),
            kernels=os.environ.get("INFERLAB_KERNELS", "numpy"),
            num_kv_blocks=_int("INFERLAB_NUM_KV_BLOCKS", 256, 4, 65536),
            block_size=_int("INFERLAB_BLOCK_SIZE", 16, 1, 256),
            max_num_seqs=_int("INFERLAB_MAX_NUM_SEQS", 8, 1, 256),
            max_num_batched_tokens=_int("INFERLAB_MAX_BATCHED_TOKENS", 256, 16, 65536),
            enable_prefix_caching=_bool("INFERLAB_PREFIX_CACHING", True),
            enable_chunked_prefill=_bool("INFERLAB_CHUNKED_PREFILL", True),
            max_waiting=_int("INFERLAB_MAX_WAITING", 64, 1, 10000),
            upstream_url=os.environ.get("INFERLAB_UPSTREAM_URL") or None,
            upstream_model=os.environ.get("INFERLAB_UPSTREAM_MODEL", "default"),
            upstream_api_key=os.environ.get("INFERLAB_UPSTREAM_API_KEY") or None,
            api_keys=keys,
            auth_disabled=_bool("INFERLAB_AUTH_DISABLED", False),
            rate_limit_rps=_float("INFERLAB_RATE_LIMIT_RPS", 5.0, 0.01, 10000),
            rate_limit_burst=_int("INFERLAB_RATE_LIMIT_BURST", 20, 1, 100000),
            max_body_bytes=_int("INFERLAB_MAX_BODY_BYTES", 1_048_576, 1024, 64 * 1024 * 1024),
            max_prompt_chars=_int("INFERLAB_MAX_PROMPT_CHARS", 32_000, 16, 1_000_000),
            max_tokens_limit=_int("INFERLAB_MAX_TOKENS", 1024, 1, 32768),
            request_timeout_s=_float("INFERLAB_REQUEST_TIMEOUT_S", 120.0, 1.0, 3600.0),
            cors_origins=_list("INFERLAB_CORS_ORIGINS"),
            enable_docs=_bool("INFERLAB_ENABLE_DOCS", False),
            enable_lab=_bool("INFERLAB_ENABLE_LAB", True),
            metrics_public=_bool("INFERLAB_METRICS_PUBLIC", False),
            hsts=_bool("INFERLAB_HSTS", False),
        )
        s.validate()
        return s

    def validate(self) -> Settings:
        if self.backend not in BACKENDS:
            raise ValueError(f"INFERLAB_BACKEND must be one of {BACKENDS}")
        if self.kernels not in ("numpy", "cpp"):
            raise ValueError("INFERLAB_KERNELS must be 'numpy' or 'cpp'")
        if self.auth_disabled and not is_loopback(self.host):
            raise ValueError(
                "refusing to start: authentication is disabled but the server binds to a non-loopback "
                "address. Set INFERLAB_API_KEYS (or INFERLAB_API_KEYS_FILE) instead."
            )
        if any(len(k) < 16 for k in self.api_keys):
            raise ValueError("API keys must be at least 16 characters long")
        if "*" in self.cors_origins:
            raise ValueError("wildcard CORS origin is not allowed with credentialed API keys")
        return self
