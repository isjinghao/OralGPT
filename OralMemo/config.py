from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = Path(__file__).resolve().parent


def load_env(path: Path = BENCH_ROOT / ".env") -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class ModelConfig:
    api_key: str
    base_url: str
    model: str


ModelRole = Literal["benchmark", "answer", "verifier"]


@dataclass(frozen=True)
class Settings:
    bench_root: Path
    dataset_json: Path
    output_root: Path
    benchmark_llm: ModelConfig
    answer_llm: ModelConfig
    verifier_llm: ModelConfig
    graph_max_edges: int

    def llm_for(self, role: ModelRole) -> ModelConfig:
        if role == "benchmark":
            return self.benchmark_llm
        if role == "answer":
            return self.answer_llm
        if role == "verifier":
            return self.verifier_llm
        raise ValueError(f"Unknown LLM role: {role}")


def _env_first(*names: str, default: str | None = None) -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    if default is not None:
        return default
    raise KeyError(names[0])


def _model_config(prefix: str) -> ModelConfig:
    """Load role-specific OpenAI-compatible settings.

    Example for prefix="BENCHMARK":
      BENCHMARK_OPENAI_API_KEY / BENCHMARK_OPENAI_BASE_URL / BENCHMARK_OPENAI_MODEL
    Falls back to generic OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL.
    """
    return ModelConfig(
        api_key=_env_first(f"{prefix}_OPENAI_API_KEY", "OPENAI_API_KEY"),
        base_url=_env_first(
            f"{prefix}_OPENAI_BASE_URL",
            "OPENAI_BASE_URL",
            default="https://api.openai.com/v1",
        ),
        model=_env_first(f"{prefix}_OPENAI_MODEL", "OPENAI_MODEL", default="qwen3.6-chat"),
    )


def get_settings() -> Settings:
    # Load both the parent workspace .env (if any) and this project .env.
    # Existing environment variables win because load_env uses os.environ.setdefault.
    load_env(ROOT / ".env")
    load_env(BENCH_ROOT / ".env")
    return Settings(
        bench_root=BENCH_ROOT,
        dataset_json=BENCH_ROOT / "oralgpt_cmf_llamafactory_sft_dataset.json",
        output_root=BENCH_ROOT / "outputs" / "group1" / "CHENFANG",
        benchmark_llm=_model_config("BENCHMARK"),
        answer_llm=_model_config("ANSWER"),
        verifier_llm=_model_config("VERIFIER"),
        graph_max_edges=int(os.environ.get("GRAPH_MAX_EDGES", "40")),
    )
