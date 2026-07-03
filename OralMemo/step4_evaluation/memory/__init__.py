"""memory 包: 统一管理各种记忆方法

- base.py                 公共基类 MemoryMethod 与共享工具 format_stage_input / collect_stage_images
- single_stage_memory.py  单阶段基线 SingleStageMemory
- full_context_memory.py  全上下文基线 FullContextMemory
- summary_memory.py       记忆基线 SummaryMemory
- mem0_memory.py          基于 mem0 的检索式记忆 Mem0Memory
"""
from __future__ import annotations

from step4_evaluation.memory.base import MemoryMethod, collect_stage_images, format_stage_input
from step4_evaluation.memory.full_context_memory import FullContextMemory
from step4_evaluation.memory.mem0_memory import Mem0Memory
from step4_evaluation.memory.single_stage_memory import SingleStageMemory
from step4_evaluation.memory.summary_memory import SummaryMemory

# 方法注册表
_REGISTRY: dict[str, type[MemoryMethod]] = {
    SingleStageMemory.name: SingleStageMemory,   # single_stage_memory
    FullContextMemory.name: FullContextMemory,   # full_context_memory
    SummaryMemory.name: SummaryMemory,           # summary_memory
    Mem0Memory.name: Mem0Memory,                 # mem0_memory
}

# 不指定 --methods 时默认只跑单阶段基线
DEFAULT_METHOD: str = SingleStageMemory.name


def available_methods() -> list[str]:
    return list(_REGISTRY)


def build_methods(names: list[str] | None = None, multimodal: bool = False) -> list[MemoryMethod]:
    selected = names or [DEFAULT_METHOD]
    return [_REGISTRY[name](multimodal=multimodal) for name in selected]


__all__ = [
    "MemoryMethod",
    "format_stage_input",
    "collect_stage_images",
    "SingleStageMemory",
    "FullContextMemory",
    "SummaryMemory",
    "Mem0Memory",
    "build_methods",
    "available_methods",
    "DEFAULT_METHOD",
]
