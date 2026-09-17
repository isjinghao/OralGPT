from __future__ import annotations

from step4_evaluation.memory.base import MemoryMethod, format_stage_input


class SingleStageMemory(MemoryMethod):

    name = "single_stage_memory"

    def __init__(self) -> None:
        super().__init__()
        self._buffer = ""

    def reset(self) -> None:
        self._buffer = ""

    def observe(self, stage: dict) -> None:

        self._buffer = format_stage_input(stage)

    def context(self, query: str | None = None) -> str:
        return self._buffer
