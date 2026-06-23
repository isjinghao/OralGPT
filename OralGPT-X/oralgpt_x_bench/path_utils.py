"""Shared path helpers for running against an external BAGEL checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def bench_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_bagel_on_path(bagel_root: str | Path | None = None) -> Path:
    root = Path(bagel_root or os.environ.get("BAGEL_ROOT", "")).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            "Set BAGEL_ROOT to your Bagel repository root "
            "(directory containing modeling/bagel/bagel.py)."
        )
    marker = root / "modeling" / "bagel" / "bagel.py"
    if not marker.is_file():
        raise FileNotFoundError(f"Invalid BAGEL_ROOT (missing {marker})")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = base or bench_root()
    return (base / path).resolve()
