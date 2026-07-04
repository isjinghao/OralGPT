"""单阶段基线: 每个阶段清空记忆, 只保留当前阶段信息
"""
from __future__ import annotations

from step4_evaluation.memory.base import MemoryMethod, collect_stage_images, format_stage_input


class SingleStageMemory(MemoryMethod):
    """单阶段基线: 每阶段清空, 只保留当前阶段。"""

    name = "single_stage_memory"

    def __init__(self, multimodal: bool = False) -> None:
        super().__init__(multimodal)
        self._buffer = ""
        self._images: list[str] = []

    def reset(self) -> None:
        self._buffer = ""
        self._images = []

    def observe(self, stage: dict) -> None:
        # 直接替换, 等价于每阶段 memory.clear()
        self._buffer = format_stage_input(stage)
        self._images = collect_stage_images(stage)

    def context(self, query: str | None = None) -> str:
        return self._buffer

    def images(self) -> list[str]:
        return list(self._images)
