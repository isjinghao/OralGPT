from __future__ import annotations

from step4_evaluation.memory.base import MemoryMethod, format_stage_input
from step4_evaluation.memory.full_context_memory import FullContextMemory
from step4_evaluation.memory.graphiti_memory import GraphitiMemory
from step4_evaluation.memory.langmem_memory import LangMemMemory
from step4_evaluation.memory.mem0_memory import Mem0Memory
from step4_evaluation.memory.single_stage_memory import SingleStageMemory
from step4_evaluation.memory.summary_memory import SummaryMemory
from step4_evaluation.memory.vector_memory import VectorMemory


_REGISTRY: dict[str, type[MemoryMethod]] = {
    SingleStageMemory.name: SingleStageMemory,   # single_stage_memory
    FullContextMemory.name: FullContextMemory,   # full_context_memory
    SummaryMemory.name: SummaryMemory,
    Mem0Memory.name: Mem0Memory,
    VectorMemory.name: VectorMemory,
    LangMemMemory.name: LangMemMemory,
    GraphitiMemory.name: GraphitiMemory,
}


DEFAULT_METHOD: str = FullContextMemory.name


def available_methods() -> list[str]:
    return list(_REGISTRY)


def build_methods(names: list[str] | None = None) -> list[MemoryMethod]:
    selected = names or [DEFAULT_METHOD]
    return [_REGISTRY[name]() for name in selected]


__all__ = [
    "MemoryMethod",
    "format_stage_input",
    "SingleStageMemory",
    "FullContextMemory",
    "SummaryMemory",
    "Mem0Memory",
    "VectorMemory",
    "LangMemMemory",
    "GraphitiMemory",
    "build_methods",
    "available_methods",
    "DEFAULT_METHOD",
]
