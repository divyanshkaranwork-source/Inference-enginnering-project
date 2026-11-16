"""Model registry, safe downloads and loading by spec.

Live models are Llama-2-architecture checkpoints small enough for a CPU.
Downloads use HTTPS only, follow redirects only to HTTPS, cap the size,
verify a pinned SHA-256 when one is known, write atomically, and record the
hash of every downloaded file (trust-on-first-use) so later tampering is
detected. Weight files are parsed as plain data - never unpickled.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from inferlab.config import TINY_RANDOM
from inferlab.foundations.model import LlamaModel
from inferlab.foundations.tokenizer import BaseTokenizer, ByteTokenizer, Llama2Tokenizer
from inferlab.foundations.weights import load_hf, load_llama2c, random_model

DEFAULT_MODELS_DIR = Path(os.environ.get("INFERLAB_MODELS_DIR", "models"))


@dataclass(frozen=True)
class Artifact:
    name: str
    url: str
    size: int | None
    sha256: str | None
    description: str


# llama2.c tokenizer pinned to a commit; TinyStories checkpoints (MIT, A. Karpathy)
ARTIFACTS: dict[str, Artifact] = {
    a.name: a
    for a in (
        Artifact(
            "tokenizer.bin",
            "https://raw.githubusercontent.com/karpathy/llama2.c/350e04fe35433e6d2941dce5a1f53308f87058eb/tokenizer.bin",
            433869,
            "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361",
            "Llama-2 SentencePiece vocabulary (32000 tokens) in llama2.c format",
        ),
        Artifact(
            "stories15M.bin",
            "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
            60816028,
            None,
            "TinyStories Llama-2 15M (dim 288, 6 layers) - fast on any laptop",
        ),
        Artifact(
            "stories42M.bin",
            "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin",
            167020572,
            None,
            "TinyStories Llama-2 42M (dim 512, 8 layers)",
        ),
        Artifact(
            "stories110M.bin",
            "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin",
            438381596,
            None,
            "TinyStories Llama-2 110M (dim 768, 12 layers) - draft with 15M for speculative decoding",
        ),
    )
}

ALLOWED_HOSTS = {"raw.githubusercontent.com", "huggingface.co", "cdn-lfs.huggingface.co",
                 "cdn-lfs-us-1.huggingface.co", "cas-bridge.xethub.hf.co"}
MAX_DOWNLOAD = 2 * 1024**3


class DownloadError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checksums_file(models_dir: Path) -> Path:
    return models_dir / "SHA256SUMS.json"


def _load_sums(models_dir: Path) -> dict[str, str]:
    p = _checksums_file(models_dir)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text("utf-8"))
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
    except (OSError, json.JSONDecodeError):
        return {}


def verify_file(name: str, models_dir: Path = DEFAULT_MODELS_DIR) -> bool:
    """True if the file matches its pinned or recorded SHA-256."""
    path = models_dir / name
    art = ARTIFACTS.get(name)
    expected = (art.sha256 if art else None) or _load_sums(models_dir).get(name)
    return path.exists() and expected is not None and _sha256(path) == expected


def download(name: str, models_dir: Path = DEFAULT_MODELS_DIR, force: bool = False, progress=None) -> Path:
    import httpx

    if name not in ARTIFACTS:
        raise DownloadError(f"unknown artifact {name!r}; choose from {sorted(ARTIFACTS)}")
    art = ARTIFACTS[name]
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / art.name
    if dest.exists() and not force:
        return dest
    try:
        return _download(art, dest, models_dir, progress)
    except httpx.HTTPError as exc:
        raise DownloadError(
            f"could not download {art.name} ({type(exc).__name__}). Check your network/proxy, or fetch "
            f"{art.url} manually into {models_dir}/"
        ) from exc


def _download(art: Artifact, dest: Path, models_dir: Path, progress) -> Path:
    from urllib.parse import urlsplit

    import httpx

    url = art.url
    for _ in range(5):  # follow redirects manually so every hop is checked
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
            raise DownloadError(f"refusing to download from {parts.scheme}://{parts.hostname}")
        with (
            httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0), follow_redirects=False) as client,
            client.stream("GET", url) as resp,
        ):
            if resp.status_code in (301, 302, 303, 307, 308):
                url = httpx.URL(url).join(resp.headers.get("location", "")).human_repr()
                continue
            if resp.status_code != 200:
                raise DownloadError(f"HTTP {resp.status_code} for {art.name}")
            h = hashlib.sha256()
            total = 0
            fd, tmp = tempfile.mkstemp(dir=models_dir, prefix=f".{art.name}.", suffix=".part")
            try:
                with os.fdopen(fd, "wb") as fh:
                    for chunk in resp.iter_bytes(1 << 20):
                        total += len(chunk)
                        if total > MAX_DOWNLOAD or (art.size and total > art.size):
                            raise DownloadError("download larger than expected")
                        h.update(chunk)
                        fh.write(chunk)
                        if progress:
                            progress(total, art.size)
                if art.size and total != art.size:
                    raise DownloadError(f"size mismatch: got {total}, expected {art.size}")
                digest = h.hexdigest()
                if art.sha256 and digest != art.sha256:
                    raise DownloadError("SHA-256 mismatch - file rejected")
                # public model data: world-readable so non-root containers can mount it read-only
                os.chmod(tmp, 0o644)
                os.replace(tmp, dest)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
            sums = _load_sums(models_dir)
            sums[art.name] = digest
            _checksums_file(models_dir).write_text(json.dumps(sums, indent=2), "utf-8")
            return dest
    raise DownloadError("too many redirects")


# ---------------------------------------------------------------------------
def resolve_model_spec(spec: str, models_dir: Path = DEFAULT_MODELS_DIR) -> str:
    """'auto' -> the best downloaded TinyStories model, else 'tiny-random'."""
    if spec != "auto":
        return spec
    for name in ("stories15M.bin", "stories42M.bin", "stories110M.bin"):
        if (models_dir / name).exists() and (models_dir / "tokenizer.bin").exists():
            return str(models_dir / name)
    return "tiny-random"


def load(spec: str = "auto", tokenizer: str | None = None, linear_kind: str = "fp32", kernels=None,
         models_dir: Path = DEFAULT_MODELS_DIR, verify: bool = True) -> tuple[LlamaModel, BaseTokenizer]:
    """Load a model + tokenizer from a spec: 'auto', 'tiny-random', a llama2.c .bin or an HF dir."""
    spec = resolve_model_spec(spec, models_dir)
    if spec == "tiny-random":
        return random_model(TINY_RANDOM, seed=0, linear_kind=linear_kind, kernels=kernels), ByteTokenizer()
    path = Path(spec)
    tok_path = Path(tokenizer) if tokenizer else models_dir / "tokenizer.bin"
    if not tok_path.is_file():
        raise FileNotFoundError(f"tokenizer not found at {tok_path}; run `inferlab download tokenizer.bin`")
    if verify:
        for p in (path, tok_path):
            recorded = p.parent.resolve() == models_dir.resolve() and p.name in _load_sums(models_dir)
            if recorded and not verify_file(p.name, models_dir):
                raise RuntimeError(f"{p.name} does not match its recorded SHA-256 - re-download it")
    if path.is_dir():
        model = load_hf(path, linear_kind, kernels)
    elif path.is_file():
        model = load_llama2c(path, linear_kind, kernels, name=path.stem)
    else:
        raise FileNotFoundError(f"model {spec!r} not found")
    tok = Llama2Tokenizer.from_file(tok_path, model.cfg.vocab_size)
    return model, tok
