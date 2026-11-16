"""Model architecture configurations.

``LlamaConfig`` describes any Llama-family decoder-only transformer. The same
class is used for:

* the *reference* model of the course, Llama-2-7B (used for all memory, FLOP,
  roofline and parallelism math - you do not need to download it), and
* the *live* models that actually run on your CPU inside this project
  (TinyStories Llama-2 checkpoints or a randomly initialised tiny Llama).

Because both share one architecture, every concept you see running on the
tiny model transfers 1:1 to Llama-2-7B - only the numbers get bigger.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LlamaConfig:
    name: str
    dim: int  # hidden size (d_model)
    n_layers: int
    n_heads: int  # query heads
    n_kv_heads: int  # key/value heads (== n_heads -> MHA, < n_heads -> GQA, 1 -> MQA)
    hidden_dim: int  # SwiGLU FFN inner size
    vocab_size: int
    max_seq_len: int
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    # "interleaved": rotate pairs (0,1),(2,3)... (Meta / llama2.c layout)
    # "half":        rotate pairs (i, i + d/2)  (Hugging Face layout)
    rope_style: str = "interleaved"
    tie_embeddings: bool = True
    chat_template: str = "plain"  # "plain" | "llama2"
    # Mixture-of-Experts (only used by the parallelism planner, e.g. Mixtral)
    n_experts: int = 0
    n_experts_active: int = 0
    extra: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if min(self.dim, self.n_layers, self.n_heads, self.n_kv_heads, self.hidden_dim) <= 0:
            raise ValueError("all model dimensions must be positive")
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads (GQA groups)")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.rope_style not in ("interleaved", "half"):
            raise ValueError("rope_style must be 'interleaved' or 'half'")
        if self.vocab_size <= 0 or self.max_seq_len <= 0:
            raise ValueError("vocab_size and max_seq_len must be positive")

    # ---- derived quantities -------------------------------------------------
    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def gqa_group(self) -> int:
        return self.n_heads // self.n_kv_heads

    def param_breakdown(self) -> dict[str, int]:
        """Exact parameter count per component (dense models)."""
        d, L, f, v = self.dim, self.n_layers, self.hidden_dim, self.vocab_size
        attn = L * (d * d + 2 * d * self.kv_dim + d * d)  # wq, wk, wv, wo
        experts = max(self.n_experts, 1)
        ffn = L * 3 * d * f * experts  # w1 (gate), w2 (down), w3 (up)
        router = L * d * self.n_experts if self.n_experts else 0
        norms = L * 2 * d + d
        embed = v * d
        lm_head = 0 if self.tie_embeddings else v * d
        return {
            "embedding": embed,
            "attention": attn,
            "ffn": ffn,
            "router": router,
            "norms": norms,
            "lm_head": lm_head,
        }

    @property
    def n_params(self) -> int:
        return sum(self.param_breakdown().values())

    @property
    def n_active_params(self) -> int:
        """Parameters touched per token (differs from n_params only for MoE)."""
        if not self.n_experts:
            return self.n_params
        b = self.param_breakdown()
        ffn_active = b["ffn"] // self.n_experts * self.n_experts_active
        return b["embedding"] + b["attention"] + ffn_active + b["router"] + b["norms"] + b["lm_head"]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("extra", None)
        d.update(
            head_dim=self.head_dim,
            kv_dim=self.kv_dim,
            gqa_group=self.gqa_group,
            n_params=self.n_params,
            n_active_params=self.n_active_params,
        )
        return d


# ---------------------------------------------------------------------------
# Reference models (math only - used by calculators, planner and docs)
# ---------------------------------------------------------------------------
LLAMA2_7B = LlamaConfig(
    name="llama-2-7b",
    dim=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=32,
    hidden_dim=11008,
    vocab_size=32000,
    max_seq_len=4096,
    rope_style="half",
    tie_embeddings=False,
    chat_template="llama2",
)

LLAMA2_13B = LlamaConfig(
    name="llama-2-13b",
    dim=5120,
    n_layers=40,
    n_heads=40,
    n_kv_heads=40,
    hidden_dim=13824,
    vocab_size=32000,
    max_seq_len=4096,
    rope_style="half",
    tie_embeddings=False,
    chat_template="llama2",
)

LLAMA2_70B = LlamaConfig(
    name="llama-2-70b",
    dim=8192,
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,
    hidden_dim=28672,
    vocab_size=32000,
    max_seq_len=4096,
    rope_style="half",
    tie_embeddings=False,
    chat_template="llama2",
)

LLAMA3_8B = LlamaConfig(
    name="llama-3-8b",
    dim=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    hidden_dim=14336,
    vocab_size=128256,
    max_seq_len=8192,
    rope_theta=500000.0,
    rope_style="half",
    tie_embeddings=False,
)

MIXTRAL_8X7B = LlamaConfig(
    name="mixtral-8x7b",
    dim=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    hidden_dim=14336,
    vocab_size=32000,
    max_seq_len=32768,
    rope_theta=1e6,
    rope_style="half",
    tie_embeddings=False,
    n_experts=8,
    n_experts_active=2,
)

# ---------------------------------------------------------------------------
# Live models (run on CPU in this project)
# ---------------------------------------------------------------------------
# TinyStories checkpoints trained by Andrej Karpathy with the Llama-2
# architecture and the Llama-2 tokenizer (32000 tokens) - the same tokenizer
# as Llama-2-7B, so tokenization lessons transfer exactly.
STORIES_15M = LlamaConfig("stories15M", 288, 6, 6, 6, 768, 32000, 256)
STORIES_42M = LlamaConfig("stories42M", 512, 8, 8, 8, 1376, 32000, 1024)
STORIES_110M = LlamaConfig("stories110M", 768, 12, 12, 12, 2048, 32000, 1024)

# Random-weight model used by tests and when no checkpoint is downloaded.
# Uses GQA (4 query heads share 2 KV heads) so the GQA code path is exercised.
TINY_RANDOM = LlamaConfig(
    name="tiny-random",
    dim=128,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,
    hidden_dim=352,  # multiple of 32 so llama.cpp-style Q8_0/Q4_0 blocks fit
    vocab_size=259,  # 3 specials + 256 bytes (byte-level tokenizer)
    max_seq_len=512,
)

REFERENCE_MODELS: dict[str, LlamaConfig] = {
    c.name: c for c in (LLAMA2_7B, LLAMA2_13B, LLAMA2_70B, LLAMA3_8B, MIXTRAL_8X7B)
}
LIVE_MODELS: dict[str, LlamaConfig] = {
    c.name: c for c in (STORIES_15M, STORIES_42M, STORIES_110M, TINY_RANDOM)
}
ALL_MODELS: dict[str, LlamaConfig] = {**REFERENCE_MODELS, **LIVE_MODELS}


def get_model_config(name: str) -> LlamaConfig:
    try:
        return ALL_MODELS[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(ALL_MODELS)}") from None
