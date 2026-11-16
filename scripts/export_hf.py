"""Export a llama2.c checkpoint (e.g. stories15M.bin) to Hugging Face safetensors.

Lets you serve the *same* tiny model with vLLM / SGLang / TensorRT-LLM and
compare them with the built-in engine:

    python scripts/export_hf.py models/stories15M.bin models/stories15M-hf
    # then copy tokenizer files from any Llama-2 repository (same 32000-token vocab), e.g.
    # tokenizer.model / tokenizer.json / tokenizer_config.json, into models/stories15M-hf/

RoPE note: llama2.c uses interleaved RoPE; HF Llama uses rotate-half, so the
q/k projection rows are permuted (exactly what HF's own conversion script does).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inferlab.foundations.weights import interleaved_to_half, load_llama2c, save_hf  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="llama2.c .bin checkpoint")
    ap.add_argument("dst", type=Path, help="output directory")
    a = ap.parse_args()
    if a.dst.exists() and any(a.dst.iterdir()):
        raise SystemExit(f"{a.dst} is not empty - refusing to overwrite")
    model = interleaved_to_half(load_llama2c(a.src))
    save_hf(a.dst, model)
    print(f"wrote {a.dst}/model.safetensors + config.json ({model.cfg.n_params / 1e6:.1f}M params)")


if __name__ == "__main__":
    main()
