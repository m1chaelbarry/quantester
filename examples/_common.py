"""Shared path helpers for example scripts (always run from the repo root)."""

from __future__ import annotations

from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
DATA_DIR = EXAMPLES_DIR / "data"


def output_dir(strategy_file: str | Path) -> Path:
    """Per-strategy output folder next to the calling script."""
    out = Path(strategy_file).resolve().parent / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out
