"""Upload a Jez release directory to a Hugging Face model repository.

    HF_TOKEN=... python -m tools.publish_hf --release releases/jez-large-1.0.0 --repo MisreadableMind/jez-large-1.0.0 [--private] [--dry-run]

Layout on the Hub: the merged ``weights/`` at the repository root (so ``from_pretrained(repo)`` and the Jez engine's
checkpoint resolution work unchanged), ``adapter/`` when the release has one, ``calibration.json``, ``training.json``
and ``README.md`` (the model card under a Hub metadata header). ``--dry-run`` prints the plan and uploads nothing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

LANGUAGES = ["en", "de", "fr", "es", "it", "pt", "ru", "uk", "ar", "hi", "zh", "ja"]


def front_matter(base: str, licence: str, licence_link: str | None, tags: list[str]) -> str:
    lines = ["---", f"license: {licence}"]
    if licence_link:
        lines.append(f"license_link: {licence_link}")
    lines += [f"base_model: {base}", "base_model_relation: finetune", "library_name: transformers",
              "language:"] + [f"  - {l}" for l in LANGUAGES] + ["tags:"] + [f"  - {t}" for t in tags] + ["---", ""]
    return "\n".join(lines)


def plan(release: Path) -> list[tuple[Path, str]]:
    """(local path, path in repo) for every file the upload sends; the weights land at the root."""
    out: list[tuple[Path, str]] = []
    weights = release / "weights"
    if not (weights / "config.json").is_file():
        raise SystemExit(f"{weights} holds no merged weights (config.json missing)")
    for f in sorted(weights.rglob("*")):
        if f.is_file():
            out.append((f, str(f.relative_to(weights))))
    adapter = release / "adapter"
    if adapter.is_dir():
        for f in sorted(adapter.rglob("*")):
            if f.is_file():
                out.append((f, f"adapter/{f.relative_to(adapter)}"))
    for name in ("calibration.json", "training.json", "MODEL_CARD.md"):
        if (release / name).is_file():
            out.append((release / name, name))
    if not (release / "MODEL_CARD.md").is_file():
        raise SystemExit(f"{release}/MODEL_CARD.md is missing; write the card before publishing")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--repo", required=True, help="<user-or-org>/<name>")
    ap.add_argument("--base", help="base model id for the card header (default: read from training/model registry)")
    ap.add_argument("--licence", default="apache-2.0", help="Hub licence identifier")
    ap.add_argument("--licence-link", default=None)
    ap.add_argument("--tags", default="jez,jev,decision-model,label-logits,classification")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    release = Path(args.release)
    files = plan(release)
    base = args.base
    if not base:
        training = json.loads((release / "training.json").read_text()) if (release / "training.json").is_file() else {}
        base = training.get("base") or json.loads((release / "weights" / "config.json").read_text()).get("_name_or_path", "unknown")
    readme = front_matter(base, args.licence, args.licence_link, args.tags.split(",")) + (release / "MODEL_CARD.md").read_text()
    total = sum(f.stat().st_size for f, _ in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB -> {args.repo} ({'private' if args.private else 'public'})")
    for f, dest in files:
        print(f"  {dest:60s} {f.stat().st_size / 1e6:10.1f} MB")
    print("  README.md (card + metadata header)")
    if args.dry_run:
        print(readme.split("---")[1])
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set")
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(f)) for f, dest in files]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode()))
    info = api.create_commit(repo_id=args.repo, operations=ops, commit_message=f"{release.name}: weights, adapter, calibration, card")
    print(f"uploaded: {info.commit_url if hasattr(info, 'commit_url') else info}")


if __name__ == "__main__":
    main()
