"""记忆方法的公共基类、接口定义与共享工具
所有记忆实现都应继承 MemoryMethod, 并实现 reset/observe/context;
update 默认空操作(仅需巩固记忆的方法才重写)
"""
from __future__ import annotations

from abc import ABC, abstractmethod


def collect_stage_images(stage: dict) -> list[str]:
    """收集一个阶段涉及的所有图片路径(阶段级 + 各 QA 级), 去重且保序。"""
    seen: list[str] = []
    for path in stage.get("image_paths", []) or []:
        if path and path not in seen:
            seen.append(path)
    for qa in stage.get("qa_pairs", []) or []:
        for path in qa.get("image_paths", []) or []:
            if path and path not in seen:
                seen.append(path)
    return seen


def format_stage_input(stage: dict, multimodal: bool = False) -> str:
    """把一个阶段的结构化数据格式化为可读文本块。

    输入:
      stage      - 轨迹中的单个 stage(含 stage_id/modality/image_paths/qa_pairs)。
      multimodal - 是否支持多模态图片输入。
                     False: 纯文本记忆, 剥除 <image> 占位, 不带入任何图片路径。
                     True : 把阶段级与逐条 QA 的图片路径带入记忆, 以便后续支持
                            image_url 的多模态大模型据此读取图片。
    输出: str - 带阶段头与逐条 Q/A 的文本; 噪声轮次带标签。
    """
    modality = ", ".join(stage.get("modality", [])) or "none"
    image_paths = collect_stage_images(stage)
    lines = [f"[Stage {stage['stage_id']} | modality: {modality} | images: {len(image_paths)}]"]
    if multimodal and image_paths:
        lines.append("Stage images: " + " | ".join(image_paths))
    for qa in stage.get("qa_pairs", []):
        human = (qa.get("human") or "").replace("<image>", "").strip()
        assistant = (qa.get("assistant") or "").strip()
        role = qa.get("role", "evidence")
        tag = "" if role == "evidence" else f" [noise:{qa.get('noise_category', role)}]"
        lines.append(f"Q{tag}: {human}")
        qa_images = qa.get("image_paths", []) or []
        if multimodal and qa_images:
            lines.append("Images: " + " | ".join(qa_images))
        lines.append(f"A: {assistant}")
    return "\n".join(lines)


class MemoryMethod(ABC):
    """记忆方法统一接口。"""

    name: str = "base"

    def __init__(self, multimodal: bool = False) -> None:
        # multimodal=True 时, 记忆中会带入图片路径, 供支持 image_url 的多模态大模型读取。
        self.multimodal = multimodal

    def setup(self, workdir) -> None:  # noqa: D401
        """可选钩子: 接收本方法专属的工作目录(如向量库持久化路径)。默认忽略。

        上层在构建方法后、reset 之前调用, 传入 <cache_root>/<name>。
        需要落盘的方法(如 mem0)据此自配置, pipeline 无需知道任何方法特有参数。
        """
        return None

    @abstractmethod
    def reset(self) -> None:
        """在一条轨迹开始前重置内部状态。"""

    @abstractmethod
    def observe(self, stage: dict) -> None:
        """观察(读取)一个阶段的信息。"""

    def update(self, llm, cache_key: str) -> None:  # noqa: D401
        """将观察到的信息巩固进记忆; 默认空操作(基线方法无需巩固)。"""
        return None

    @abstractmethod
    def context(self, query: str | None = None) -> str:
        """返回当前用于回答问题的上下文文本。

        query - 检索式记忆(如 mem0)据此做相关性检索; 非检索方法可忽略。
        """

    def images(self) -> list[str]:
        """返回当前记忆中(有序去重的)图片路径; 纯文本方法返回空列表。

        多模态评测时, 上层会据此把图片转成 image_url 分块附加给大模型。
        """
        return []
