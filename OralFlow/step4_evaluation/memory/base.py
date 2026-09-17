from __future__ import annotations

from abc import ABC, abstractmethod
import time
from pathlib import Path
from threading import Lock


def normalize_query(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(part for item in value if (part := normalize_query(item)))
    if isinstance(value, dict):
        for key in ("text", "content"):
            if key in value:
                return normalize_query(value[key])
    return str(value).strip()


def format_stage_input(stage: dict) -> str:

    modality = ", ".join(stage.get("modality", [])) or "none"
    lines = []
    for qa in stage["qa_pairs"]:
        role = qa["role"]
        if role == "evaluation":
            continue
        if role != "observation":
            raise ValueError(f"Unsupported QA role in trajectory: {role}")
        human = (qa.get("human") or "").replace("<image>", "").strip()
        assistant = (qa.get("assistant") or "").strip()
        noise_category = qa.get("noise_category")
        tag = f" [noise:{noise_category}]" if noise_category else ""
        lines.append(f"Q{tag}: {human}")
        lines.append(f"A: {assistant}")
    if not lines:
        return ""
    header = f"[Stage {stage['stage_id']} | modality: {modality}]"
    return "\n".join([header] + lines)


class MemoryMethod(ABC):

    name: str = "base"

    def __init__(self) -> None:
        self.workdir: Path | None = None
        self.namespace = ""
        self._metrics_lock = Lock()
        self._metrics = {
            "write_calls": 0,
            "write_seconds": 0.0,
            "retrieval_calls": 0,
            "retrieval_seconds": 0.0,
            "failures": 0,
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "embedding_calls": 0,
            "embedding_tokens": 0,
        }

    def setup(self, workdir, namespace: str = "") -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def observe(self, stage: dict) -> None:
        pass

    def update(self, llm, cache_key: str) -> None:
        return None

    def add_metrics(self, **values: int | float) -> None:
        with self._metrics_lock:
            for key, value in values.items():
                self._metrics[key] += value

    def restore_metrics(self, values: dict) -> None:
        with self._metrics_lock:
            for key in self._metrics:
                self._metrics[key] = values.get(key, 0)

    def write(self, stage: dict, llm, cache_key: str) -> None:
        started = time.perf_counter()
        self.add_metrics(write_calls=1)
        try:
            self.observe(stage)
            self.update(llm, cache_key)
        except Exception:
            self.add_metrics(failures=1)
            raise
        finally:
            self.add_metrics(write_seconds=time.perf_counter() - started)

    @abstractmethod
    def context(self, query: str | None = None) -> str:
        pass

    def timed_context(self, query: str | None = None) -> str:
        started = time.perf_counter()
        self.add_metrics(retrieval_calls=1)
        try:
            return self.context(query)
        except Exception:
            self.add_metrics(failures=1)
            raise
        finally:
            self.add_metrics(retrieval_seconds=time.perf_counter() - started)

    def metrics(self) -> dict:
        with self._metrics_lock:
            result = dict(self._metrics)
        result["write_seconds"] = round(result["write_seconds"], 6)
        result["write_avg_seconds"] = round(
            result["write_seconds"] / result["write_calls"], 6
        ) if result["write_calls"] else 0.0
        result["retrieval_seconds"] = round(result["retrieval_seconds"], 6)
        result["retrieval_avg_seconds"] = round(
            result["retrieval_seconds"] / result["retrieval_calls"], 6
        ) if result["retrieval_calls"] else 0.0
        operations = max(result["write_calls"] + result["retrieval_calls"], result["failures"])
        result["failure_rate"] = round(result["failures"] / operations, 6) if operations else 0.0
        return result

    def close(self) -> None:
        return None
