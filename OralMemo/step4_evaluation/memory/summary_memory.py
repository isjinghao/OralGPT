"""记忆基线: 增量把每阶段信息由大模型融入一份紧凑的结构化记忆。
"""
from __future__ import annotations
import os

from step4_evaluation.memory.base import MemoryMethod, format_stage_input
from step4_evaluation.templating import render


SUMMARY_MEMORY_SCHEMA = {
    "type": "object",
    "properties": {"memory": {"type": "string"}},
    "required": ["memory"],
    "additionalProperties": False,
}


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(text for item in value if (text := _to_text(item)))
    if isinstance(value, dict):
        return "\n".join(
            text for key in ("text", "content", "memory")
            if key in value and (text := _to_text(value[key]))
        )
    return str(value).strip()


class SummaryMemory(MemoryMethod):
    """记忆基线: 增量把每阶段融入一份紧凑结构化记忆。"""

    name = "summary_memory"

    def __init__(self) -> None:
        super().__init__()
        self._summary = ""
        self._pending = ""

    def reset(self) -> None:
        self._summary = ""
        self._pending = ""

    def observe(self, stage: dict) -> None:
        self._pending = format_stage_input(stage)

    def update(self, llm, cache_key: str) -> None:
        if not self._pending:
            return
        prompt = render(
            "memory_update",
            existing_memory=self._summary or "(empty)",
            new_stage=self._pending,
        )
        before = llm.client.usage_snapshot()
        try:
            data = llm.complete(
                prompt,
                cache_key=cache_key,
                max_tokens=int(os.environ.get("SUMMARY_MEMORY_MAX_TOKENS", "4096")),
                required_keys=("memory",),
                json_schema_name="summary_memory_update",
                json_schema=SUMMARY_MEMORY_SCHEMA,
            )
        finally:
            after = llm.client.usage_snapshot()
            self.add_metrics(
                llm_calls=after["calls"] - before["calls"],
                input_tokens=after["input_tokens"] - before["input_tokens"],
                output_tokens=after["output_tokens"] - before["output_tokens"],
            )
        memory = _to_text(data.get("memory"))
        if not memory:
            raise ValueError("Summary memory update returned empty memory")
        self._summary = memory
        self._pending = ""

    def context(self, query: str | None = None) -> str:
        return self._summary
