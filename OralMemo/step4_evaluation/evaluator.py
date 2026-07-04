"""Step4 流式评测引擎: 缓存 LLM 封装、按阶段流式读取轨迹并在问题释放时刻提问"""
from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

from step4_evaluation.memory import MemoryMethod
from step4_evaluation.templating import render


class CachedLLM:
    """对 ChatClient 的缓存封装: 相同 cache_key 直接读磁盘, 避免重复调用。"""

    def __init__(self, client, cache_dir: Path) -> None:
        self.client = client
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.hits = 0

    def complete(self, prompt: str, cache_key: str, max_tokens: int = 4000,
                 temperature: float = 0.0, images: list[str] | None = None) -> dict:
        path = self.cache_dir / f"{cache_key}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))
        data = self.client.complete_json(prompt, temperature=temperature, max_tokens=max_tokens, images=images)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.calls += 1
        return data


def encode_image(path: Path) -> str | None:
    # 把本地图片读为 data URL(base64); 文件不存在时返回 None
    if not path.exists() or not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def gather_image_urls(method: MemoryMethod, image_root: Path) -> list[str]:
    # 把记忆中的图片路径转成可传给大模型的 data URL 列表
    urls: list[str] = []
    for rel in method.images():
        url = encode_image(image_root / rel)
        if url:
            urls.append(url)
    return urls


def answer_question(method: MemoryMethod, task: dict, llm: CachedLLM, image_root: Path | None = None) -> dict:
    """基于记忆方法当前上下文回答一个任务的问题, 返回作答记录。

    method.multimodal=True 且提供 image_root 时, 会把记忆中的图片以 image_url 分块附带给大模型。
    """
    prompt = render(
        "answer",
        memory=method.context(task["question"]) or "(empty)",
        question=task["question"],
    )
    images: list[str] | None = None
    if method.multimodal and image_root is not None:
        images = gather_image_urls(method, image_root) or None

    data = llm.complete(prompt, cache_key=f"answer_{task['task_id']}", max_tokens=4000, images=images)
    answer = (data.get("answer") or "").strip()
    return {
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "ask_after_stage": task.get("ask_after_stage"),
        "question": task["question"],
        "gold_answer": task.get("gold_answer", ""),
        "model_answer": answer,
        "selected_evidence": task.get("selected_evidence", []),
        "validation_accepted": task.get("validation", {}).get("accepted", True),
        "n_images": len(images) if images else 0,
    }


def run_streaming(
    method: MemoryMethod,
    trajectory: dict,
    tasks_by_stage: dict[str, list[dict]],
    llm: CachedLLM,
    image_root: Path | None = None,
) -> list[dict]:
    """按阶段顺序流式读取轨迹, 在每个阶段结束后释放并回答该阶段的问题。

    输入:
      method         - 记忆方法实例
      trajectory     - 单条轨迹(含 stages)
      tasks_by_stage - 按 ask_after_stage 分组的任务
      llm            - 缓存 LLM
      image_root     - 图片相对路径的根目录(多模态时用于定位并编码图片)
    输出: list[dict] - 每个任务的作答记录。
    """
    method.reset()
    records: list[dict] = []
    stages = sorted(trajectory["stages"], key=lambda s: s["order"])
    present = {s["stage_id"] for s in stages}

    for stage in stages:
        stage_id = stage["stage_id"]
        method.observe(stage)
        method.update(llm, cache_key=f"memupdate_{stage_id}")
        for task in tasks_by_stage.get(stage_id, []):
            print(f"  [{method.name}] answer {task['task_id']} @ {stage_id}", flush=True)
            records.append(answer_question(method, task, llm, image_root))

    # 轨迹缺失某阶段(变体轨迹)时, 其任务在最后一个阶段后释放
    for stage_id, tasks in tasks_by_stage.items():
        if stage_id in present:
            continue
        for task in tasks:
            print(f"  [{method.name}] answer {task['task_id']} @ END (stage {stage_id} absent)", flush=True)
            records.append(answer_question(method, task, llm, image_root))

    return records
