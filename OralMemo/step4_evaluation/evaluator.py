"""Step4 流式评测引擎: 缓存 LLM 封装、按阶段流式读取轨迹并在问题释放时刻提问"""
from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path

from batch_utils import log
from step4_evaluation.memory import MemoryMethod
from step4_evaluation.templating import render


class CachedLLM:
    def __init__(self, client, cache_dir: Path) -> None:
        self.client = client
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.hits = 0

    def complete(self, prompt: str, cache_key: str, max_tokens: int = 8000,
                 temperature: float = 0.0, images: list[str] | None = None) -> dict:
        path = self.cache_dir / f"{cache_key}.json"
        cache_input = {
            "type": "json",
            "model": self.client.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "images": images or [],
        }
        cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if cached and cached.get("input") == cache_input:
            self.hits += 1
            self.client.log("step4/cache", f"cache_hit key={cache_key}")
            return cached["result"]
        result = self.client.complete_json(
            prompt, temperature=temperature, max_tokens=max_tokens, images=images
        )
        path.write_text(
            json.dumps({"input": cache_input, "result": result}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.calls += 1
        return result

    def complete_text(self, prompt: str, cache_key: str, max_tokens: int = 8000,
                      temperature: float = 0.0, images: list[str] | None = None) -> str:
        path = self.cache_dir / f"{cache_key}.json"
        cache_input = {
            "type": "text",
            "model": self.client.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "images": images or [],
        }
        cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if cached and cached.get("input") == cache_input:
            self.hits += 1
            self.client.log("step4/cache", f"cache_hit key={cache_key}")
            return cached["answer"]
        answer = self.client.complete_text(
            prompt, temperature=temperature, max_tokens=max_tokens, images=images
        )
        path.write_text(
            json.dumps({"input": cache_input, "answer": answer}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.calls += 1
        return answer


def encode_image(path: Path) -> str | None:
    # 把本地图片读为 data URL(base64); 文件不存在时返回 None
    if not path.exists() or not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def gather_image_urls(method: MemoryMethod, image_root: Path) -> list[str]:
    # 把记忆中的图片路径转成 data URL 列表
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

    answer = llm.complete_text(prompt, cache_key=f"answer_{task['task_id']}", max_tokens=16000, images=images)
    return {
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "ask_after_stage": task.get("ask_after_stage"),
        "question": task["question"],
        "gold_answer": task.get("gold_answer", ""),
        "model_answer": answer,
        "selected_evidence": task.get("selected_evidence", []),
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
        use_gold = record["task_type"] == "treatment" and release_treatment_ground_truth
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
        "stage_type": records[0]["task_type"],
        "modality": ["TEXT_QA"],
        "image_paths": [],
        "qa_pairs": qa_pairs,
    }
    method.observe(release_stage)
    method.update(llm, cache_key=f"memupdate_{release_stage['stage_id']}")


def _stage_order(stage_id: str) -> int:
    match = re.search(r"\d+", stage_id)
    if match is None:
        raise ValueError(f"Stage id has no numeric order: {stage_id}")
    return int(match.group())


def run_streaming(
    method: MemoryMethod,
    trajectory: dict,
    tasks_by_stage: dict[str, list[dict]],
    llm: CachedLLM,
    image_root: Path | None = None,
    *,
    release_treatment_ground_truth: bool = True,
    log_prefix: str | None = None,
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
    prefix = log_prefix or f"[evaluation][{trajectory['patient_id']}]"
    records: list[dict] = []
    pending_releases: dict[str, list[dict]] = {}
    stages = sorted(trajectory["stages"], key=lambda s: s["order"])
    present = {s["stage_id"] for s in stages}
    missing_task_stages = sorted(
        (stage_id for stage_id in tasks_by_stage if stage_id not in present),
        key=_stage_order,
    )
    answered_missing: set[str] = set()

    def answer_tasks(stage_id: str, missing_stage: bool = False) -> None:
        for task in tasks_by_stage.get(stage_id, []):
            detail = f" missing_stage={stage_id}" if missing_stage else ""
            log(
                f"{prefix}[step4/answer] method={method.name} task={task['task_id']} "
                f"stage={stage_id}{detail}"
            )
            record = answer_question(method, task, llm, image_root)
            records.append(record)
            release_after_stage = task.get("release_after_stage")
            if release_after_stage is not None:
                record["release_after_stage"] = release_after_stage
                pending_releases.setdefault(release_after_stage, []).append(record)

    def release_missing_before(order: int) -> None:
        release_stages = sorted(
            (
                stage_id for stage_id in pending_releases
                if stage_id not in present and _stage_order(stage_id) < order
            ),
            key=_stage_order,
        )
        for release_stage in release_stages:
            _release_task_answers(
                method,
                pending_releases.pop(release_stage),
                llm,
                release_stage,
                release_treatment_ground_truth,
            )

    for stage in stages:
        stage_id = stage["stage_id"]
        current_order = _stage_order(stage_id)
        for missing_stage in missing_task_stages:
            if missing_stage not in answered_missing and _stage_order(missing_stage) < current_order:
                answer_tasks(missing_stage, missing_stage=True)
                answered_missing.add(missing_stage)
        release_missing_before(current_order)
        method.observe(stage)
        method.update(llm, cache_key=f"memupdate_{stage_id}")
        _release_task_answers(
            method,
            pending_releases.pop(stage_id, []),
            llm,
            stage_id,
            release_treatment_ground_truth,
        )
        answer_tasks(stage_id)
        _release_task_answers(
            method,
            pending_releases.pop(stage_id, []),
            llm,
            stage_id,
            release_treatment_ground_truth,
        )

    for missing_stage in missing_task_stages:
        if missing_stage not in answered_missing:
            answer_tasks(missing_stage, missing_stage=True)
    release_missing_before(10**9)
    if pending_releases:
        raise ValueError(f"Unreached evaluation release stages: {sorted(pending_releases)}")
    return records
