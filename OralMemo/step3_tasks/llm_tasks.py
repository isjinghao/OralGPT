from __future__ import annotations

import json
from pathlib import Path
from string import Template

import yaml

from bench.llm_client import ChatClient
from bench.step3_tasks.selectors import (
    EvidenceIndex,
    compact_evidence_text,
    edges_text,
    evidence_catalog,
    stages_summary,
)


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def load_template(name: str) -> Template:
    # 加载prompt 模板
    config = yaml.safe_load((PROMPT_DIR / name).read_text(encoding="utf-8"))
    return Template(config["template"])


def load_normal_task_plan() -> list[dict]:
    # 加载普通任务规划配置
    config = yaml.safe_load((PROMPT_DIR / "normal_task_plan.yaml").read_text(encoding="utf-8"))
    return config["task_types"]


def cached_completion(client: ChatClient, prompt: str, cache_path: Path, max_tokens: int) -> dict:
    # 带缓存的 LLM JSON 调用
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    result = client.complete_json(prompt, max_tokens=max_tokens)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def plan_normal_tasks(client: ChatClient, patient_stages: dict, index: EvidenceIndex, cache_dir: Path) -> list[dict]:
    # 按 normal_task_plan.yaml 逐个任务类型让 LLM 规划指定数量的任务
    patient_id = patient_stages["patient_id"]
    prefix = patient_id.replace("__", "_")
    template = load_template("task_planning.yaml")
    stages_text = stages_summary(patient_stages)
    evidence_text = evidence_catalog(index)
    graph_text = edges_text(index)

    planned = []
    for entry in load_normal_task_plan():
        task_type = entry["task_type"]
        cache_path = cache_dir / "task_planning" / f"{prefix}_{task_type}.json"
        prompt = template.substitute(
            patient_id=patient_id,
            task_type=task_type,
            task_count=entry["count"],
            type_instruction=entry["instruction"],
            stages_text=stages_text,
            evidence_text=evidence_text,
            edges_text=graph_text,
        )
        result = cached_completion(client, prompt, cache_path, max_tokens=8000)
        planned.extend(result["tasks"])
    return planned


def generate_question(client: ChatClient, spec: dict, cache_dir: Path) -> dict:
    # 依据任务规格让 LLM 生成自然临床问题与临床链, 结果缓存到 question_generation/
    cache_path = cache_dir / "question_generation" / f"{spec['task_id']}.json"
    template = load_template("question_generation.yaml")
    prompt = template.substitute(
        patient_id=spec["patient_id"],
        task_type=spec["task_type"],
        ask_after_stage=spec["ask_after_stage"],
        gold_answer=spec["gold_answer"],
        evidence_text=compact_evidence_text([
            {
                "evidence_id": item["evidence_id"],
                "introduced_stage": item["stage"],
                "modality": item["modality"],
                "fact_text": item["fact_text"],
            }
            for item in spec["selected_evidence"]
        ]),
    )
    return cached_completion(client, prompt, cache_path, max_tokens=4000)


def validate_task(client: ChatClient, task: dict, cache_dir: Path) -> dict:
    # 让 LLM 判断问题/金标准是否合规(是否泄题、能否在指定阶段后回答等), 结果缓存到 qa_validation/
    cache_path = cache_dir / "qa_validation" / f"{task['task_id']}.json"
    template = load_template("qa_validation.yaml")
    task_for_prompt = {
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "ask_after_stage": task["ask_after_stage"],
        "question": task["question"],
        "gold_answer": task["gold_answer"],
        "required_evidence_ids": [item["evidence_id"] for item in task["selected_evidence"]],
        "selected_evidence": task["selected_evidence"],
    }
    prompt = template.substitute(task_json=json.dumps(task_for_prompt, ensure_ascii=False, indent=2))
    return cached_completion(client, prompt, cache_path, max_tokens=4000)


def finalize_task(client: ChatClient, spec: dict, cache_dir: Path) -> dict:
    # 先生成问题再校验; 若未通过且给出修正问题则采用修正问题, 返回带 validation 的最终任务
    
    # (1) 生成问题
    candidate = generate_question(client, spec, cache_dir)
    task = dict(spec)
    task["question"] = candidate["question"]
    task["clinical_chain"] = candidate.get("clinical_chain", "")
    
    # (2) 问题验证
    validation = validate_task(client, task, cache_dir)
    if not validation.get("accepted") and validation.get("fixed_question"):
        task["question"] = validation["fixed_question"]
    task["validation"] = validation
    return task


def select_heldout_evidence(client: ChatClient, task_id: str, question: str, answer: str, index: EvidenceIndex, cache_dir: Path) -> list[str]:
    # 让 LLM 基于问题、标准答案、全部证据目录与证据图, 选出该 QA 真正依赖的 evidence_id 子集, 结果缓存到 heldout_evidence/
    cache_path = cache_dir / "heldout_evidence" / f"{task_id}.json"
    template = load_template("evidence_selection.yaml")
    prompt = template.substitute(
        question=question,
        answer=answer,
        evidence_text=evidence_catalog(index),
        edges_text=edges_text(index),
    )
    result = cached_completion(client, prompt, cache_path, max_tokens=12000)
    return result["required_evidence_ids"]


def generate_rubric(client: ChatClient, task: dict, cache_dir: Path) -> dict:
    # 为诊断/治疗任务生成评分 rubric, 结果缓存到 rubric_generation/。
    cache_path = cache_dir / "rubric_generation" / f"{task['task_id']}.json"
    template = load_template("rubric_generation.yaml")
    prompt = template.substitute(
        task_type=task["task_type"],
        question=task["question"],
        answer=task["gold_answer"],
        evidence_text=compact_evidence_text([
            {
                "evidence_id": item["evidence_id"],
                "introduced_stage": item["stage"],
                "modality": item["modality"],
                "fact_text": item["fact_text"],
            }
            for item in task["selected_evidence"]
        ]),
    )
    result = cached_completion(client, prompt, cache_path, max_tokens=12000)
    return {
        "rubric_id": f"{task['task_type']}_{task['task_id']}",
        "task_id": task["task_id"],
        "max_score": result.get("max_score", 100),
        "criteria": result["criteria"],
    }
