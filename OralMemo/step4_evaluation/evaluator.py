"""Step4 流式评测引擎: 缓存 LLM 封装、按阶段流式读取轨迹并在问题释放时刻提问"""
from __future__ import annotations
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Semaphore

from utils.batch_utils import log
from utils.image_utils import image_data_url
from utils.json_utils import write_json_atomic
from step4_evaluation.memory import MemoryMethod
from step4_evaluation.templating import render


class CachedLLM:
    def __init__(self, client, cache_dir: Path) -> None:
        self.client = client
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_cache(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def complete(self, prompt: str, cache_key: str, max_tokens: int = 4096,
                 temperature: float = 0.0, images: list[str] | None = None,
                 timeout: int = 300) -> dict:
        path = self.cache_dir / f"{cache_key}.json"
        cache_input = {
            "type": "json",
            "model": self.client.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "images": images or [],
        }
        cached = self._load_cache(path)
        if cached and cached.get("input") == cache_input:
            return cached["result"]
        result = self.client.complete_json(
            prompt, temperature=temperature, max_tokens=max_tokens, images=images, timeout=timeout
        )
        write_json_atomic(path, {"input": cache_input, "result": result})
        return result

    def complete_text(self, prompt: str, cache_key: str, max_tokens: int = 4096,
                      temperature: float = 0.0, images: list[str] | None = None,
                      timeout: int = 300) -> str:
        path = self.cache_dir / f"{cache_key}.json"
        cache_input = {
            "type": "text",
            "model": self.client.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "images": images or [],
        }
        cached = self._load_cache(path)
        if cached and cached.get("input") == cache_input:
            return cached["answer"]
        answer = self.client.complete_text(
            prompt, temperature=temperature, max_tokens=max_tokens, images=images, timeout=timeout
        )
        write_json_atomic(path, {"input": cache_input, "answer": answer})
        return answer


def gather_image_urls(
    method: MemoryMethod,
    image_root: Path,
    image_cache: dict[Path, str | None],
) -> list[str]:
    # 同一轨迹内跨问题、跨记忆方法复用图片 data URL，避免重复读取和编码。
    urls: list[str] = []
    for rel in method.images():
        path = image_root / rel
        if path not in image_cache:
            image_cache[path] = image_data_url(path)
        url = image_cache[path]
        if url:
            urls.append(url)
    return urls


def answer_question(
    method: MemoryMethod,
    task: dict,
    llm: CachedLLM,
    image_root: Path | None,
    image_cache: dict[Path, str | None],
) -> dict:
    """基于记忆方法当前上下文回答一个任务的问题, 返回作答记录。

    method.multimodal=True 且提供 image_root 时, 会把记忆中的图片以 image_url 分块附带给大模型。
    """
    memory_context = method.timed_context(task["question"]) or "(empty)"
    prompt = render(
        "answer",
        memory=memory_context,
        question=task["question"],
    )
    images: list[str] | None = None
    if method.multimodal and image_root is not None:
        images = gather_image_urls(method, image_root, image_cache) or None

    max_tokens = 4096 if task["task_type"] == "treatment" else 2048
    answer = llm.complete_text(
        prompt, cache_key=f"answer_{task['task_id']}", max_tokens=max_tokens, images=images
    )
    return {
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "ask_after_stage": task.get("ask_after_stage"),
        "question": task["question"],
        "gold_answer": task.get("gold_answer", ""),
        "model_answer": answer,
        "memory_context": memory_context,
        "selected_evidence": task.get("selected_evidence", []),
        "n_images": len(images) if images else 0,
    }


def _release_task_answers(
    method: MemoryMethod,
    records: list[dict],
    memory_llm: CachedLLM,
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
        "order": _stage_order(stage_id),
        "stage_type": records[0]["task_type"],
        "modality": ["TEXT_QA"],
        "image_paths": [],
        "qa_pairs": qa_pairs,
    }
    method.write(
        release_stage,
        memory_llm,
        cache_key=f"memupdate_{release_stage['stage_id']}",
    )



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
    image_root: Path | None,
    *,
    image_cache: dict[Path, str | None],
    answer_semaphore: Semaphore,
    answer_workers: int = 2,
    memory_llm: CachedLLM,
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
        tasks = tasks_by_stage.get(stage_id, [])
        detail = f" missing_stage={stage_id}" if missing_stage else ""

        def answer(task: dict) -> dict:
            log(
                f"{prefix}[step4/answer] method={method.name} task={task['task_id']} "
                f"stage={stage_id}{detail}"
            )
            with answer_semaphore:
                return answer_question(method, task, llm, image_root, image_cache)

        with ThreadPoolExecutor(max_workers=answer_workers) as executor:
            stage_records = list(executor.map(answer, tasks))
        records.extend(stage_records)
        for task, record in zip(tasks, stage_records):
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
                memory_llm,
                release_stage,
                release_treatment_ground_truth,
            )

    def answer_missing_before(order: int) -> None:
        for missing_stage in missing_task_stages:
            missing_order = _stage_order(missing_stage)
            if missing_stage in answered_missing or missing_order >= order:
                continue
            release_missing_before(missing_order)
            _release_task_answers(
                method,
                pending_releases.pop(missing_stage, []),
                memory_llm,
                missing_stage,
                release_treatment_ground_truth,
            )
            answer_tasks(missing_stage, missing_stage=True)
            answered_missing.add(missing_stage)
            _release_task_answers(
                method,
                pending_releases.pop(missing_stage, []),
                memory_llm,
                missing_stage,
                release_treatment_ground_truth,
            )

    for stage in stages:
        stage_id = stage["stage_id"]
        current_order = _stage_order(stage_id)
        answer_missing_before(current_order)
        release_missing_before(current_order)
        method.write(stage, memory_llm, cache_key=f"memupdate_{stage_id}")
        _release_task_answers(
            method,
            pending_releases.pop(stage_id, []),
            memory_llm,
            stage_id,
            release_treatment_ground_truth,
        )
        answer_tasks(stage_id)
        _release_task_answers(
            method,
            pending_releases.pop(stage_id, []),
            memory_llm,
            stage_id,
            release_treatment_ground_truth,
        )

    answer_missing_before(10**9)
    release_missing_before(10**9)
    if pending_releases:
        raise ValueError(f"Unreached evaluation release stages: {sorted(pending_releases)}")
    return records
