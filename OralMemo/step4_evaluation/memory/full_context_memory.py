"""全上下文基线: 把所有历史阶段原样拼接进上下文，非记忆方法"""
from __future__ import annotations

from step4_evaluation.memory.base import MemoryMethod, format_stage_input


class FullContextMemory(MemoryMethod):
    """全上下文基线: 追加所有历史阶段原文。"""

    name = "full_context_memory"

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def reset(self) -> None:
        self._chunks = []

    def observe(self, stage: dict) -> None:
        self._chunks.append(format_stage_input(stage))

    def context(self, query: str | None = None) -> str:
        return "\n\n".join(self._chunks)
