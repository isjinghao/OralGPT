from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = Path(__file__).resolve().parent


def load_env(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    root: Path
    bench_root: Path
    dataset_json: Path
    data_root: Path
    output_root: Path
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    graph_max_edges: int


def get_settings() -> Settings:
    load_env()
    return Settings(
        root=ROOT,
        bench_root=BENCH_ROOT,
        dataset_json=BENCH_ROOT / "oralgpt_cmf_llamafactory_sft_dataset.json",
        data_root=BENCH_ROOT / "SH9HCMFdata",
        output_root=BENCH_ROOT / "outputs" / "group1" / "CHENFANG",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        openai_model=os.environ.get("OPENAI_MODEL", "qwen3.6-chat"),
        graph_max_edges=int(os.environ.get("GRAPH_MAX_EDGES", "25")),
    )
