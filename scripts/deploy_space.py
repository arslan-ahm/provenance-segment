"""Publish the explorer to a Hugging Face Space (Docker SDK) with the `hf` CLI.

Stages exactly the committed files the image needs (Dockerfile, project metadata, src/, runs/)
plus space/README.md as the Space card, then uploads them in one commit and removes files
that no longer belong. Needs `hf auth login` locally, or HF_TOKEN in CI.

    uv run python scripts/deploy_space.py --space ARslan-Ahamd/provenance-segment
    uv run python scripts/deploy_space.py --space ARslan-Ahamd/provenance-segment --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ("Dockerfile", "pyproject.toml", "uv.lock", "LICENSE", ".gitattributes", "src/", "runs/")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout


def staged_files() -> list[str]:
    files = [f for f in git("ls-files", "-z").split("\0") if f]
    return sorted(f for f in files if f in INCLUDE or any(p.endswith("/") and f.startswith(p) for p in INCLUDE))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--space", required=True, help="owner/name of the Space")
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded and stop")
    ap.add_argument("--allow-dirty", action="store_true", help="deploy even if tracked files have uncommitted changes")
    args = ap.parse_args()

    dirty = git("status", "--porcelain", "--untracked-files=no", "--", *INCLUDE, "space/").strip()
    if dirty and not args.allow_dirty:
        print(f"Uncommitted changes in deployed paths; commit first or pass --allow-dirty:\n{dirty}", file=sys.stderr)
        return 1
    files = staged_files()
    commit = git("rev-parse", "--short", "HEAD").strip()
    size = sum((ROOT / f).stat().st_size for f in files)
    print(f"{len(files)} files, {size / 1e6:.1f} MB, from commit {commit} -> spaces/{args.space}")
    if args.dry_run:
        print("\n".join(["README.md  (from space/README.md)", *files]))
        return 0
    if shutil.which("hf") is None:
        print("The `hf` CLI is not on PATH (uv tool install huggingface_hub).", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for f in files:
            (out / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, out / f)
        shutil.copy2(ROOT / "space" / "README.md", out / "README.md")
        subprocess.run(["hf", "repos", "create", args.space, "--type", "space", "--sdk", "docker", "--public", "--exist-ok"], check=True)
        subprocess.run(
            ["hf", "upload", args.space, str(out), ".", "--repo-type", "space", "--delete", "*", "--commit-message", f"Deploy {commit}"],
            check=True,
        )
    print(f"https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
