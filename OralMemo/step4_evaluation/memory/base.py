"""记忆方法的公共基类、接口定义与共享工具
所有记忆实现都应继承 MemoryMethod, 并实现 reset/observe/context;
update 默认空操作(仅需巩固记忆的方法才重写)
"""
from __future__ import annotations

from abc import ABC, abstractmethod


def collect_stage_images(stage: dict) -> list[str]:
    #返回阶段级图片路径。
    return [path for path in (stage.get("image_paths", []) or []) if path]

def format_stage_input(stage: dict) -> str:
    # 把一个阶段的结构化数据格式化为可读文本块
    modality = ", ".join(stage.get("modality", [])) or "none"
    image_paths = collect_stage_images(stage)
    lines = [f"[Stage {stage['stage_id']} | modality: {modality} | images: {len(image_paths)}]"]
    for qa in stage.get("qa_pairs", []):
        human = (qa.get("human") or "").replace("<image>", "").strip()
        assistant = (qa.get("assistant") or "").strip()
        role = qa.get("role", "evidence")
        tag = "" if role == "evidence" else f" [noise:{qa.get('noise_category', role)}]"
        lines.append(f"Q{tag}: {human}")
        lines.append(f"A: {assistant}")
    return "\n".join(lines)


class MemoryMethod(ABC):
    """记忆方法统一接口。"""

    name: str = "base"

    def __init__(self, multimodal: bool = False) -> None:
        self.multimodal = multimodal

    def setup(self, workdir) -> None:
        """可选钩子，接收本方法专属的工作目录(如向量库持久化路径)
        """
        return None

    @abstractmethod
    def reset(self) -> None:
        """在一条轨迹开始前重置内部状态。"""

    @abstractmethod
    def observe(self, stage: dict) -> None:
        """读取一个阶段的信息。"""

    def update(self, llm, cache_key: str) -> None:
        """将观察到的信息更新进记忆"""
        return None

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
