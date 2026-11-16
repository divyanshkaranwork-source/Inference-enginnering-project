"""Loading and saving Llama weights - *safely*.

Supported formats

* **llama2.c ``.bin``** - a 7 x int32 header followed by raw float32 arrays.
  Used by the TinyStories checkpoints (stories15M / 42M / 110M).
* **Hugging Face ``safetensors``** (+ ``config.json``) - used by
  Llama-2-7B, TinyLlama, etc. Parsed with a strict, dependency-free reader.
* **random** - deterministic random init for tests / offline demos.

Security notes (see SECURITY.md):

* We never use ``pickle`` / ``torch.load``: a ``.pt`` / ``.bin`` pickle file
  can execute arbitrary code when loaded. Both formats here are *pure data*.
* Every header field is bounds-checked and the file size must match the size
  implied by the header exactly, so a malformed or truncated file is rejected
  before any array is created.
* Sharded safetensors indexes may only reference plain file names inside the
  model directory (no path traversal).
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np

from inferlab.config import LlamaConfig
from inferlab.foundations.model import LayerWeights, LlamaModel, LlamaWeights
from inferlab.foundations.quantization import Linear, make_linear

F32 = np.float32

# Sanity limits for untrusted headers (generous: Llama-3-405B fits)
MAX_DIM = 65536
MAX_LAYERS = 512
MAX_VOCAB = 1_000_000
MAX_SEQ = 1_048_576
MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024


class WeightFormatError(ValueError):
    """Raised when a weight file is malformed or inconsistent."""


def _check_range(name: str, value: int, lo: int, hi: int) -> int:
    if not lo <= value <= hi:
        raise WeightFormatError(f"{name}={value} outside allowed range [{lo}, {hi}]")
    return value


# ---------------------------------------------------------------------------
# Random weights
# ---------------------------------------------------------------------------
def random_weights(cfg: LlamaConfig, seed: int = 0, linear_kind: str = "fp32") -> LlamaWeights:
    rng = np.random.default_rng(seed)
    d, f, kv, std = cfg.dim, cfg.hidden_dim, cfg.kv_dim, 0.02
    out_std = std / np.sqrt(2 * cfg.n_layers)  # GPT-2 style scaled init for residual outputs

    def lin(o, i, s=std):
        return make_linear(rng.normal(0, s, (o, i)).astype(F32), linear_kind)

    layers = [
        LayerWeights(
            attn_norm=np.ones(d, F32),
            wq=lin(d, d),
            wk=lin(kv, d),
            wv=lin(kv, d),
            wo=lin(d, d, out_std),
            ffn_norm=np.ones(d, F32),
            w1=lin(f, d),
            w2=lin(d, f, out_std),
            w3=lin(f, d),
        )
        for _ in range(cfg.n_layers)
    ]
    emb = rng.normal(0, 1.0, (cfg.vocab_size, d)).astype(F32)
    return LlamaWeights(emb, layers, np.ones(d, F32), Linear(emb), tied=True)


def random_model(cfg: LlamaConfig, seed: int = 0, linear_kind: str = "fp32", kernels=None):
    return LlamaModel(cfg, random_weights(cfg, seed, linear_kind), kernels)


# ---------------------------------------------------------------------------
# llama2.c format
# ---------------------------------------------------------------------------
_HEADER = struct.Struct("<7i")


def _llama2c_layout(cfg: LlamaConfig, shared: bool) -> list[tuple[str, tuple[int, ...]]]:
    L, d, f, v, kv = cfg.n_layers, cfg.dim, cfg.hidden_dim, cfg.vocab_size, cfg.kv_dim
    hd = cfg.head_dim
    layout = [
        ("tok_embeddings", (v, d)),
        ("attn_norm", (L, d)),
        ("wq", (L, d, d)),
        ("wk", (L, kv, d)),
        ("wv", (L, kv, d)),
        ("wo", (L, d, d)),
        ("ffn_norm", (L, d)),
        ("w1", (L, f, d)),
        ("w2", (L, d, f)),
        ("w3", (L, f, d)),
        ("final_norm", (d,)),
        ("freq_cis_real", (cfg.max_seq_len, hd // 2)),  # legacy, unused (we recompute RoPE)
        ("freq_cis_imag", (cfg.max_seq_len, hd // 2)),
    ]
    if not shared:
        layout.append(("lm_head", (v, d)))
    return layout


def read_llama2c_header(path: str | os.PathLike, name: str | None = None) -> tuple[LlamaConfig, bool]:
    path = Path(path)
    with path.open("rb") as fh:
        raw = fh.read(_HEADER.size)
    if len(raw) != _HEADER.size:
        raise WeightFormatError("file too small for llama2.c header")
    dim, hidden, n_layers, n_heads, n_kv, vocab, seq = _HEADER.unpack(raw)
    shared = vocab > 0  # llama2.c convention: negative vocab = separate classifier
    vocab = abs(vocab)
    _check_range("dim", dim, 1, MAX_DIM)
    _check_range("hidden_dim", hidden, 1, 4 * MAX_DIM)
    _check_range("n_layers", n_layers, 1, MAX_LAYERS)
    _check_range("n_heads", n_heads, 1, dim)
    _check_range("n_kv_heads", n_kv, 1, n_heads)
    _check_range("vocab_size", vocab, 1, MAX_VOCAB)
    _check_range("seq_len", seq, 1, MAX_SEQ)
    try:
        cfg = LlamaConfig(
            name=name or path.stem,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv,
            hidden_dim=hidden,
            vocab_size=vocab,
            max_seq_len=seq,
            rope_style="interleaved",
            tie_embeddings=shared,
        )
    except ValueError as exc:
        raise WeightFormatError(f"inconsistent header: {exc}") from exc
    return cfg, shared


def load_llama2c(
    path: str | os.PathLike, linear_kind: str = "fp32", kernels=None, name: str | None = None
) -> LlamaModel:
    """Memory-map a llama2.c checkpoint (no copy for fp32: pages load lazily)."""
    path = Path(path)
    cfg, shared = read_llama2c_header(path, name)
    layout = _llama2c_layout(cfg, shared)
    n_floats = sum(int(np.prod(s)) for _, s in layout)
    expected = _HEADER.size + 4 * n_floats
    actual = path.stat().st_size
    if actual != expected:
        raise WeightFormatError(f"file size {actual} != expected {expected} bytes for this header")

    mm = np.memmap(path, dtype=F32, mode="r", offset=_HEADER.size, shape=(n_floats,))
    arrays: dict[str, np.ndarray] = {}
    off = 0
    for key, shape in layout:
        n = int(np.prod(shape))
        arrays[key] = mm[off : off + n].reshape(shape)
        off += n
    return _assemble(cfg, arrays, linear_kind, kernels, layered=True)


def save_llama2c(path: str | os.PathLike, model: LlamaModel) -> None:
    """Write a model in llama2.c format (used by tests and ``inferlab export``)."""
    cfg, w = model.cfg, model.w
    vocab = cfg.vocab_size if w.tied else -cfg.vocab_size
    with Path(path).open("wb") as fh:
        fh.write(
            _HEADER.pack(cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, vocab, cfg.max_seq_len)
        )

        def put(a):
            fh.write(np.ascontiguousarray(a, dtype=F32).tobytes())

        put(w.tok_embeddings)
        put(np.stack([lw.attn_norm for lw in w.layers]))
        for attr in ("wq", "wk", "wv", "wo"):
            put(np.stack([getattr(lw, attr).dequantize() for lw in w.layers]))
        put(np.stack([lw.ffn_norm for lw in w.layers]))
        for attr in ("w1", "w2", "w3"):
            put(np.stack([getattr(lw, attr).dequantize() for lw in w.layers]))
        put(w.final_norm)
        put(np.zeros((cfg.max_seq_len, cfg.head_dim), F32))  # freq_cis real+imag placeholders
        if not w.tied:
            put(w.lm_head.dequantize())


def _assemble(cfg, arrays, linear_kind, kernels, layered: bool) -> LlamaModel:
    def lin(a):
        return make_linear(a, linear_kind) if linear_kind != "fp32" else Linear(a)

    layers = []
    for i in range(cfg.n_layers):
        g = (lambda k, i=i: arrays[k][i]) if layered else (lambda k, i=i: arrays[f"{k}.{i}"])
        layers.append(
            LayerWeights(
                attn_norm=np.asarray(g("attn_norm"), F32),
                wq=lin(g("wq")),
                wk=lin(g("wk")),
                wv=lin(g("wv")),
                wo=lin(g("wo")),
                ffn_norm=np.asarray(g("ffn_norm"), F32),
                w1=lin(g("w1")),
                w2=lin(g("w2")),
                w3=lin(g("w3")),
            )
        )
    emb = arrays["tok_embeddings"]
    tied = "lm_head" not in arrays
    head = Linear(emb) if tied else lin(arrays["lm_head"])
    w = LlamaWeights(emb, layers, np.asarray(arrays["final_norm"], F32), head, tied=tied)
    return LlamaModel(cfg, w, kernels)


# ---------------------------------------------------------------------------
# safetensors (Hugging Face)
# ---------------------------------------------------------------------------
_ST_DTYPES = {"F32": (np.float32, 4), "F16": (np.float16, 2), "BF16": (np.uint16, 2)}


def read_safetensors(path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Strict safetensors reader returning float32 arrays (memory-mapped when F32).

    Format: 8-byte little-endian header length N, N bytes of JSON, raw data.
    """
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise WeightFormatError("file too small for safetensors header")
        (n,) = struct.unpack("<Q", raw)
        if n == 0 or n > MAX_SAFETENSORS_HEADER or 8 + n > size:
            raise WeightFormatError("invalid safetensors header length")
        try:
            header = json.loads(fh.read(n).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WeightFormatError("safetensors header is not valid JSON") from exc
    if not isinstance(header, dict):
        raise WeightFormatError("safetensors header must be an object")
    data_start, data_len = 8 + n, size - 8 - n
    out: dict[str, np.ndarray] = {}
    spans = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict) or meta.get("dtype") not in _ST_DTYPES:
            raise WeightFormatError(f"tensor {name!r}: unsupported or missing dtype")
        shape, offs = meta.get("shape"), meta.get("data_offsets")
        if (
            not isinstance(shape, list)
            or not all(isinstance(s, int) and 0 <= s <= MAX_VOCAB * MAX_DIM for s in shape)
            or not isinstance(offs, list)
            or len(offs) != 2
            or not all(isinstance(o, int) for o in offs)
        ):
            raise WeightFormatError(f"tensor {name!r}: malformed shape/offsets")
        np_dtype, itemsize = _ST_DTYPES[meta["dtype"]]
        begin, end = offs
        count = int(np.prod(shape)) if shape else 1
        if not 0 <= begin <= end <= data_len or end - begin != count * itemsize:
            raise WeightFormatError(f"tensor {name!r}: offsets out of bounds or size mismatch")
        spans.append((begin, end))
        arr = np.memmap(path, dtype=np_dtype, mode="r", offset=data_start + begin, shape=(count,))
        if meta["dtype"] == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(F32)
        elif meta["dtype"] == "F16":
            arr = arr.astype(F32)
        out[name] = arr.reshape(shape)
    spans.sort()
    for (_b0, e0), (b1, _) in zip(spans, spans[1:], strict=False):
        if b1 < e0:
            raise WeightFormatError("overlapping tensors in safetensors file")
    return out


def write_safetensors(path: str | os.PathLike, tensors: dict[str, np.ndarray]) -> None:
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=F32)
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
        blobs.append(a.tobytes())
        off += a.nbytes
    hb = json.dumps(header).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)  # align data to 8 bytes
    with Path(path).open("wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for b in blobs:
            fh.write(b)


def _safe_child(directory: Path, filename: str) -> Path:
    if not isinstance(filename, str) or Path(filename).name != filename or filename in ("", ".", ".."):
        raise WeightFormatError(f"illegal shard file name {filename!r}")
    p = (directory / filename).resolve()
    if p.parent != directory.resolve():
        raise WeightFormatError("shard path escapes the model directory")
    return p


def hf_config_to_llama(cfg_json: dict, name: str) -> LlamaConfig:
    def geti(key, default=None):
        v = cfg_json.get(key, default)
        if not isinstance(v, int) or isinstance(v, bool):
            raise WeightFormatError(f"config.json: {key} must be an integer")
        return v

    d = _check_range("hidden_size", geti("hidden_size"), 1, MAX_DIM)
    h = geti("num_attention_heads")
    return LlamaConfig(
        name=name,
        dim=d,
        n_layers=_check_range("num_hidden_layers", geti("num_hidden_layers"), 1, MAX_LAYERS),
        n_heads=_check_range("num_attention_heads", h, 1, d),
        n_kv_heads=_check_range("num_key_value_heads", geti("num_key_value_heads", h), 1, h),
        hidden_dim=_check_range("intermediate_size", geti("intermediate_size"), 1, 4 * MAX_DIM),
        vocab_size=_check_range("vocab_size", geti("vocab_size"), 1, MAX_VOCAB),
        max_seq_len=_check_range(
            "max_position_embeddings", geti("max_position_embeddings", 2048), 1, MAX_SEQ
        ),
        rope_theta=float(cfg_json.get("rope_theta", 10000.0)),
        norm_eps=float(cfg_json.get("rms_norm_eps", 1e-5)),
        rope_style="half",
        tie_embeddings=bool(cfg_json.get("tie_word_embeddings", False)),
        chat_template="llama2" if "chat" in name.lower() else "plain",
    )


HF_NAMES = {
    "attn_norm": "model.layers.{i}.input_layernorm.weight",
    "wq": "model.layers.{i}.self_attn.q_proj.weight",
    "wk": "model.layers.{i}.self_attn.k_proj.weight",
    "wv": "model.layers.{i}.self_attn.v_proj.weight",
    "wo": "model.layers.{i}.self_attn.o_proj.weight",
    "ffn_norm": "model.layers.{i}.post_attention_layernorm.weight",
    "w1": "model.layers.{i}.mlp.gate_proj.weight",
    "w2": "model.layers.{i}.mlp.down_proj.weight",
    "w3": "model.layers.{i}.mlp.up_proj.weight",
}


def load_hf(directory: str | os.PathLike, linear_kind: str = "fp32", kernels=None) -> LlamaModel:
    """Load a Hugging Face Llama checkpoint directory (config.json + *.safetensors).

    Tip: Llama-2-7B is ~27 GB in fp32. Use ``linear_kind="int8"`` (~7 GB) or
    ``"q4_0"`` (~4 GB) on a laptop - and enjoy discovering why real engines
    exist: NumPy decode on 7B is a few seconds per token.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise WeightFormatError(f"{directory} is not a directory")
    try:
        cfg_json = json.loads((directory / "config.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WeightFormatError("missing or invalid config.json") from exc
    if not isinstance(cfg_json, dict):
        raise WeightFormatError("config.json must be an object")
    cfg = hf_config_to_llama(cfg_json, directory.name)

    index = directory / "model.safetensors.index.json"
    if index.exists():
        try:
            weight_map = json.loads(index.read_text("utf-8"))["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise WeightFormatError("invalid safetensors index") from exc
        files = sorted({_safe_child(directory, f) for f in weight_map.values()})
    else:
        files = [_safe_child(directory, "model.safetensors")]
    tensors: dict[str, np.ndarray] = {}
    for f in files:
        tensors.update(read_safetensors(f))

    def need(key):
        if key not in tensors:
            raise WeightFormatError(f"missing tensor {key}")
        return tensors[key]

    arrays: dict[str, np.ndarray] = {
        "tok_embeddings": need("model.embed_tokens.weight"),
        "final_norm": need("model.norm.weight"),
    }
    for i in range(cfg.n_layers):
        for short, pattern in HF_NAMES.items():
            arrays[f"{short}.{i}"] = need(pattern.format(i=i))
    if not cfg.tie_embeddings and "lm_head.weight" in tensors:
        arrays["lm_head"] = tensors["lm_head.weight"]
    return _assemble(cfg, arrays, linear_kind, kernels, layered=False)


def save_hf(directory: str | os.PathLike, model: LlamaModel) -> None:
    """Export to HF layout (rope_style must be 'half'; see ``interleaved_to_half``)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cfg, w = model.cfg, model.w
    tensors = {"model.embed_tokens.weight": w.tok_embeddings, "model.norm.weight": w.final_norm}
    for i, lw in enumerate(w.layers):
        for short, pattern in HF_NAMES.items():
            obj = getattr(lw, short)
            tensors[pattern.format(i=i)] = obj if isinstance(obj, np.ndarray) else obj.dequantize()
    if not w.tied:
        tensors["lm_head.weight"] = w.lm_head.dequantize()
    write_safetensors(directory / "model.safetensors", tensors)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": cfg.dim,
                "num_hidden_layers": cfg.n_layers,
                "num_attention_heads": cfg.n_heads,
                "num_key_value_heads": cfg.n_kv_heads,
                "intermediate_size": cfg.hidden_dim,
                "vocab_size": cfg.vocab_size,
                "max_position_embeddings": cfg.max_seq_len,
                "rope_theta": cfg.rope_theta,
                "rms_norm_eps": cfg.norm_eps,
                "tie_word_embeddings": w.tied,
            },
            indent=2,
        ),
        "utf-8",
    )


def interleaved_to_half(model: LlamaModel) -> LlamaModel:
    """Permute wq/wk rows so an interleaved-RoPE model runs with half-RoPE.

    This is the permutation Hugging Face's ``convert_llama_weights_to_hf.py``
    applies to Meta's original checkpoints. Within each head, output row
    2j -> j and 2j+1 -> j + hd/2.
    """
    from dataclasses import replace

    cfg = model.cfg
    hd = cfg.head_dim
    perm_head = np.concatenate([np.arange(0, hd, 2), np.arange(1, hd, 2)])

    def permute(lin, n_heads):
        W = lin.dequantize().reshape(n_heads, hd, -1)[:, perm_head, :]
        return Linear(np.ascontiguousarray(W.reshape(n_heads * hd, -1)))

    layers = [
        LayerWeights(
            lw.attn_norm,
            permute(lw.wq, cfg.n_heads),
            permute(lw.wk, cfg.n_kv_heads),
            lw.wv,
            lw.wo,
            lw.ffn_norm,
            lw.w1,
            lw.w2,
            lw.w3,
        )
        for lw in model.w.layers
    ]
    w = LlamaWeights(model.w.tok_embeddings, layers, model.w.final_norm, model.w.lm_head, model.w.tied)
    return LlamaModel(replace(cfg, rope_style="half"), w, model.kernels)
