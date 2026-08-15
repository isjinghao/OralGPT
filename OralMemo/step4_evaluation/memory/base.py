"""记忆方法的公共基类、接口定义与共享工具
所有记忆实现都应继承 MemoryMethod, 并实现 reset/observe/context;
update 默认空操作(仅需巩固记忆的方法才重写)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import time
from pathlib import Path
from threading import Lock


def collect_stage_images(stage: dict) -> list[str]:
    #返回阶段级图片路径。
    return [path for path in (stage.get("image_paths", []) or []) if path]

def format_stage_input(stage: dict) -> str:
    # 把一个阶段的结构化数据格式化为可读文本块
    modality = ", ".join(stage.get("modality", [])) or "none"
    image_paths = collect_stage_images(stage)
    lines = [f"[Stage {stage['stage_id']} | modality: {modality} | images: {len(image_paths)}]"]
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
    return "\n".join(lines)


class MemoryMethod(ABC):
    """记忆方法统一接口。"""

    name: str = "base"

    def __init__(self, multimodal: bool = False) -> None:
        self.multimodal = multimodal
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
        """在一条轨迹开始前重置内部状态。"""

    @abstractmethod
    def observe(self, stage: dict) -> None:
        """读取一个阶段的信息。"""

    def update(self, llm, cache_key: str) -> None:
        """将观察到的信息更新进记忆"""
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
        """返回当前用于回答问题的上下文文本。

        query - 检索式记忆(如 mem0)据此做相关性检索; 非检索方法可忽略。
        """

    def images(self) -> list[str]:
        """返回当前记忆中(有序去重的)图片路径; 纯文本方法返回空列表。

        多模态评测时, 上层会据此把图片转成 image_url 分块附加给大模型
        """
        return []

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
