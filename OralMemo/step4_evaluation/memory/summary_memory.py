"""记忆基线: 增量把每阶段信息由大模型融入一份紧凑的结构化记忆。
"""
from __future__ import annotations

from step4_evaluation.memory.base import MemoryMethod, collect_stage_images, format_stage_input
from step4_evaluation.templating import render


class SummaryMemory(MemoryMethod):
    """记忆基线: 增量把每阶段融入一份紧凑结构化记忆。"""

    name = "summary_memory"

    def __init__(self, multimodal: bool = False) -> None:
        super().__init__(multimodal)
        self._summary = ""
        self._pending = ""
        self._images: list[str] = []

    def reset(self) -> None:
        self._summary = ""
        self._pending = ""
        self._images = []

    def observe(self, stage: dict) -> None:
        self._pending = format_stage_input(stage)
        # 图片路径独立累积, 不经 LLM 文本巩固而丢失
        for path in collect_stage_images(stage):
            if path not in self._images:
                self._images.append(path)

    def update(self, llm, cache_key: str) -> None:
        if not self._pending:
            return
        prompt = render(
            "memory_update",
            existing_memory=self._summary or "(empty)",
            new_stage=self._pending,
        )
        data = llm.complete(prompt, cache_key=cache_key, max_tokens=16000)
        self._summary = (data.get("memory") or self._summary).strip()
        self._pending = ""

    def context(self, query: str | None = None) -> str:
        return self._summary

    def images(self) -> list[str]:
        return list(self._images)
