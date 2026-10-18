"""Merge a LoRA adapter into its base checkpoint and write plain weights the engine can serve.

    .venv/bin/python -m train.merge_adapter --model jez-large-1.0.0 --release releases/jez-large-1.0.0

train.finetune saves the LoRA recipe as <release>/adapter. The engine loads plain checkpoints only, so this step
loads the base text decoder, merges the adapter into it and writes <release>/weights with the tokenizer. The merged
model answers exactly as the adapter-wrapped one did; peft is not needed at serving time.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from jez import models as registry
from jez.engine.torch_backend import load_text_model

log = logging.getLogger("merge")


def merge(base: str, adapter: Path, out: Path, dtype: torch.dtype = torch.bfloat16, device: str | None = None) -> Path:
    from peft import PeftModel
    from transformers import AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, family, _ = load_text_model(base, dtype, device)
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    log.info("merged %s + %s (%s) -> %s", base, adapter, family, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="jez-large-1.0.0", help="registry id; picks the base checkpoint")
    ap.add_argument("--base", help="override the base checkpoint")
    ap.add_argument("--release", required=True, help="release directory holding adapter/")
    ap.add_argument("--adapter", help="adapter directory (default <release>/adapter)")
    ap.add_argument("--out", help="output directory (default <release>/weights)")
    ap.add_argument("--device", help="cuda or cpu (default: cuda when available)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rel = Path(args.release)
    spec = registry.MODELS[registry.resolve(args.model) or args.model]
    merge(args.base or spec.base, Path(args.adapter) if args.adapter else rel / "adapter",
          Path(args.out) if args.out else rel / "weights", device=args.device)


if __name__ == "__main__":
    main()
