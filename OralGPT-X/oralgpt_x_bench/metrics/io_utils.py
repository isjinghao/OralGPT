"""JSONL helpers with resume support."""

from __future__ import annotations

import json
import threading
from pathlib import Path

_lock = threading.Lock()


def load_completed_ids(jsonl_path: Path, id_key: str = "id") -> set[str]:
    completed: set[str] = set()
    if not jsonl_path.is_file():
        return completed
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            completed.add(str(row[id_key]))
    return completed


def append_jsonl(jsonl_path: Path, row: dict) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
