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

    def complete(self, prompt: str, cache_key: str, max_tokens: int = 8000,
                 temperature: float = 0.0, images: list[str] | None = None) -> dict:
        path = self.cache_dir / f"{cache_key}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))
        data = self.client.complete_json(prompt, temperature=temperature, max_tokens=max_tokens, images=images)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.calls += 1
        return data

    def complete_text(self, prompt: str, cache_key: str, max_tokens: int = 8000,
                      temperature: float = 0.0, images: list[str] | None = None) -> str:
        # 纯文本作答的缓存封装; 缓存沿用 {"answer": ...} 格式以兼容既有缓存
        path = self.cache_dir / f"{cache_key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")).get("answer", "")
        text = self.client.complete_text(prompt, temperature=temperature, max_tokens=max_tokens, images=images)
        path.write_text(json.dumps({"answer": text}, ensure_ascii=False, indent=2), encoding="utf-8")
        self.calls += 1
        return text


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

    answer = llm.complete_text(prompt, cache_key=f"answer_{task['task_id']}", max_tokens=16000, images=images).strip()
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


def _release_task_answers(
    method: MemoryMethod,
    records: list[dict],
    llm: CachedLLM,
    stage_id: str,
    release_treatment_ground_truth: bool,
) -> None:
    if not records:
        return
    qa_pairs = []
    for record in records:
        use_gold = (
            record.get("release_group") == "treatment"
            and release_treatment_ground_truth
        )
        qa_pairs.append(
            {
                "source_turn_id": record["task_id"],
                "human": record["question"],
                "assistant": record["gold_answer"] if use_gold else record["model_answer"],
                "image_paths": [],
                "role": "observation",

            }
        )
    release_stage = {
        "stage_id": f"{stage_id}__evaluation_release",
        "order": 0,
        "stage_type": "treatment" if records[0].get("release_group") == "treatment" else "followup",
        "modality": ["TEXT_QA"],
        "image_paths": [],
        "qa_pairs": qa_pairs,
    }
    method.observe(release_stage)
    method.update(llm, cache_key=f"memupdate_{release_stage['stage_id']}")


def run_streaming(
    method: MemoryMethod,
    trajectory: dict,
    tasks_by_stage: dict[str, list[dict]],
    llm: CachedLLM,
    image_root: Path | None = None,
    *,
    release_treatment_ground_truth: bool = True,
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
    pending_releases: dict[str, list[dict]] = {}
    stages = sorted(trajectory["stages"], key=lambda s: s["order"])
    present = {s["stage_id"] for s in stages}

    for stage in stages:
        stage_id = stage["stage_id"]
        method.observe(stage)
        method.update(llm, cache_key=f"memupdate_{stage_id}")
        # 早期已完成的治疗批次在其 release stage 开始时进入 memory，
        # 使同阶段的后续 benchmark 任务能够使用真实/模型治疗历史。
        _release_task_answers(
            method,
            pending_releases.pop(stage_id, []),
            llm,
            stage_id,
            release_treatment_ground_truth,
        )
        stage_records: list[dict] = []
        for task in tasks_by_stage.get(stage_id, []):

            print(f"  [{method.name}] answer {task['task_id']} @ {stage_id}", flush=True)
            record = answer_question(method, task, llm, image_root)
            record.update(
                {
                    "release_to_memory": bool(task.get("release_to_memory")),
                    "release_after_stage": task.get("release_after_stage"),
                    "release_group": task.get("release_group"),
                }
            )
            records.append(record)
            stage_records.append(record)
        for record in stage_records:
            if record["release_to_memory"]:
                pending_releases.setdefault(record["release_after_stage"], []).append(record)
        _release_task_answers(
            method,
            pending_releases.pop(stage_id, []),
            llm,
            stage_id,
            release_treatment_ground_truth,
        )


    if pending_releases:
        raise ValueError(f"Unreached evaluation release stages: {sorted(pending_releases)}")

    # 轨迹缺失某阶段(变体轨迹)时, 其任务在最后一个阶段后释放
    for stage_id, tasks in tasks_by_stage.items():

        if stage_id in present:
            continue
        for task in tasks:
            print(f"  [{method.name}] answer {task['task_id']} @ END (stage {stage_id} absent)", flush=True)
            records.append(answer_question(method, task, llm, image_root))

    return records
