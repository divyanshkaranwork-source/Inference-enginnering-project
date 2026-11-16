"""Shared helpers for the lab scripts (run them from the repository root)."""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # allow `python labs/01_...py` without installing
    sys.path.insert(0, str(ROOT))

from inferlab import models  # noqa: E402


def load(spec: str | None = None, **kw):
    """Load the live model: $INFERLAB_MODEL, else stories15M if downloaded, else tiny-random."""
    spec = spec or os.environ.get("INFERLAB_MODEL", "auto")
    model, tok = models.load(spec, models_dir=Path(os.environ.get("INFERLAB_MODELS_DIR", ROOT / "models")), **kw)
    print(f"[model] {model.cfg.name}: {model.cfg.n_params / 1e6:.1f}M params, {model.nbytes / 2**20:.1f} MiB")
    if model.cfg.name == "tiny-random":
        print("[model] random weights - run `inferlab download` for real text (mechanics are identical)")
    return model, tok


def header(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def explain(text: str) -> None:
    print(textwrap.fill(" ".join(text.split()), width=78, initial_indent="  ", subsequent_indent="  "))


def table(rows: list[list], headers: list[str]) -> None:
    cols = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cols) for i in range(len(headers))]
    for j, r in enumerate(cols):
        print("  " + "  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths, strict=True))))
        if j == 0:
            print("  " + "  ".join("-" * w for w in widths))
