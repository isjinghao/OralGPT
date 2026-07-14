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


def question_feedback_block(validation: dict | None) -> str:
    # 把上一轮校验的问题作为反馈, 指导模型重新生成一道合规的问题
    if not validation:
        return ""
    lines = ["===== REVIEWER FEEDBACK ON YOUR PREVIOUS QUESTION (fix ALL of these) ====="]
    if validation.get("feedback"):
        lines.append(f"- overall: {validation['feedback']}")
    for issue in validation.get("issues", []) or []:
        if isinstance(issue, dict):
            lines.append(
                f"- [{issue.get('severity', '?')}] {issue.get('problem', issue)}"
                f" | fix: {issue.get('suggested_fix', '')}"
            )
        else:
            lines.append(f"- {issue}")
    lines.append("Generate a NEW question that resolves every issue above.")
    return "\n".join(lines)


def generate_question(client: ChatClient, spec: dict, cache_dir: Path,
                      feedback: dict | None = None, attempt: int = 1,
                      gold_answer: str | None = None) -> dict:
    # 依据任务规格让 LLM 生成自然临床问题与临床链, 结果缓存到 question_generation/
    # feedback 为上一轮校验结果, attempt 用于区分多轮生成的缓存, gold_answer 允许使用修复后的金标准
    cache_path = cache_dir / "question_generation" / f"{spec['task_id']}_a{attempt}.json"
    template = load_template("question_generation.yaml")
    prompt = template.substitute(
        patient_id=spec["patient_id"],
        task_type=spec["task_type"],
        ask_after_stage=spec["ask_after_stage"],
        gold_answer=gold_answer if gold_answer is not None else spec["gold_answer"],
        evidence_text=compact_evidence_text([
            {
                "evidence_id": item["evidence_id"],
                "introduced_stage": item["stage"],
                "modality": item["modality"],
                "fact_text": item["fact_text"],
            }
            for item in spec["selected_evidence"]
        ]),
        feedback_block=question_feedback_block(feedback),
    )
    return cached_completion(client, prompt, cache_path, max_tokens=8000)


def validate_task(client: ChatClient, task: dict, cache_dir: Path, attempt: int = 1) -> dict:
    # 让 LLM 判断问题/金标准是否合规(是否泄题、能否在指定阶段后回答等), 结果缓存到 qa_validation/
    cache_path = cache_dir / "qa_validation" / f"{task['task_id']}_a{attempt}.json"
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
    return cached_completion(client, prompt, cache_path, max_tokens=8000)


def finalize_task(client: ChatClient, spec: dict, cache_dir: Path, max_iters: int = 3) -> dict:
    # 生成<->校验反馈循环: 若未通过校验, 把反馈交回模型重生成问题, 并应用校验给出的修正问题/修正金标准, 直到通过或达到最大轮数
    task = dict(spec)
    feedback: dict | None = None
    validation: dict = {}
    history: list[dict] = []
    current_gold = spec["gold_answer"]
    for it in range(1, max_iters + 1):
        # (1) 生成问题(可带上一轮反馈, 使用当前(可能已修复的)金标准)
        candidate = generate_question(client, spec, cache_dir, feedback=feedback,
                                      attempt=it, gold_answer=current_gold)
        task["question"] = candidate["question"]
        task["clinical_chain"] = candidate.get("clinical_chain", "")
        task["gold_answer"] = current_gold

        # (2) 问题+金标准验证
        validation = validate_task(client, task, cache_dir, attempt=it)
        history.append({"iteration": it, "accepted": validation.get("accepted")})
        print(f"  [loop {it}/{max_iters}] accepted={validation.get('accepted')}", flush=True)
        if validation.get("accepted"):
            # 校验通过时也应用其给出的修正(如修剪后的金标准/问题)
            if validation.get("fixed_answer"):
                current_gold = validation["fixed_answer"]
            if validation.get("fixed_question"):
                task["question"] = validation["fixed_question"]
            break

        # (3) 未通过: 应用可用的修正并把反馈交回下一轮重新生成/复验
        if validation.get("fixed_answer"):
            current_gold = validation["fixed_answer"]
        if validation.get("fixed_question"):
            task["question"] = validation["fixed_question"]
        feedback = validation

    task["gold_answer"] = current_gold
    task["validation"] = validation
    task["validation_history"] = history
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
    # rubric 只保留评分标准(name/score/description); 证据在任务里已有, 不在 rubric 中冗余存储
    criteria = [
        {
            "name": c["name"],
            "score": c["score"],
            "description": c.get("description", ""),
        }
        for c in result["criteria"]
    ]
    return {
        "rubric_id": f"{task['task_type']}_{task['task_id']}",
        "task_id": task["task_id"],
        "max_score": result.get("max_score", 100),
        "criteria": criteria,
    }
