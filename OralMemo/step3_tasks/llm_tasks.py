from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from string import Template

import yaml

from batch_utils import log
from llm_client import ChatClient
from step3_tasks.selectors import (
    EvidenceIndex,
    edges_text,
    evidence_catalog,
    human_stage_label,
    question_evidence_text,
    stages_summary,
)


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def load_template(prompt_dir: Path, name: str) -> Template:
    config = yaml.safe_load((prompt_dir / name).read_text(encoding="utf-8"))
    return Template(config["template"])


def load_normal_task_plan(prompt_dir: Path) -> list[dict]:
    config = yaml.safe_load((prompt_dir / "normal_task_plan.yaml").read_text(encoding="utf-8"))
    return config["task_types"]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def cached_completion(client: ChatClient, prompt: str, cache_path: Path, max_tokens: int) -> dict:
    cache_input = {"model": client.model, "prompt": prompt, "max_tokens": max_tokens}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("input") == cache_input:
            client.log("step3/cache", f"cache_hit file={cache_path.name}")
            return cached["result"]
    result = client.complete_json(prompt, max_tokens=max_tokens)
    _write_json(cache_path, {"input": cache_input, "result": result})
    return result


def plan_task_candidates(
    client: ChatClient,
    patient_stages: dict,
    index: EvidenceIndex,
    cache_dir: Path,
    prompt_dir: Path,
    entry: dict,
    count: int,
    attempt: int,
    feedback: list[str],
) -> list[dict]:
    patient_id = patient_stages["patient_id"]
    task_type = entry["task_type"]
    prompt = load_template(prompt_dir, "task_planning.yaml").substitute(
        patient_id=patient_id,
        task_type=task_type,
        task_count=count,
        type_instruction=entry["instruction"],
        stages_text=stages_summary(patient_stages),
        evidence_text=evidence_catalog(index),
        edges_text=edges_text(index),
    )
    if feedback:
        prompt += (
            "\n\nReviewer feedback from rejected candidates; fix these issues in the replacements:\n- "
            + "\n- ".join(feedback)
        )
    prefix = patient_id.replace("__", "_")
    cache_path = cache_dir / "task_planning" / f"{prefix}_{task_type}_a{attempt}.json"
    result = cached_completion(client, prompt, cache_path, max_tokens=8000)
    return result["tasks"]


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


def generate_question(
    client: ChatClient,
    spec: dict,
    cache_dir: Path,
    prompt_dir: Path,
    feedback: dict | None = None,
    attempt: int = 1,
    gold_answer: str | None = None,
) -> dict:
    # 依据任务规格让 LLM 生成自然临床问题与临床链, 结果缓存到 question_generation/
    # feedback 为上一轮校验结果, attempt 用于区分多轮生成的缓存, gold_answer 允许使用修复后的金标准
    cache_path = cache_dir / "question_generation" / f"{spec['task_id']}_a{attempt}.json"
    template = load_template(prompt_dir, "question_generation.yaml")
    prompt = template.substitute(
        patient_id=spec["patient_id"],
        task_type=spec["task_type"],
        ask_after_stage=human_stage_label(spec["ask_after_stage"]),
        gold_answer=gold_answer if gold_answer is not None else spec["gold_answer"],
        evidence_text=question_evidence_text([
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


def evidence_payload(evidence: list[dict]) -> list[dict]:
    return [
        {
            "evidence_id": item["evidence_id"],
            "stage": item["introduced_stage"],
            "modality": item.get("modality", []),
            "fact_text": item["fact_text"],
            "field": item.get("normalized", {}).get("field"),
            "value": item.get("normalized", {}).get("value"),
            "unit": item.get("normalized", {}).get("unit"),
            "tooth": item.get("normalized", {}).get("tooth"),
            "side": item.get("normalized", {}).get("side"),
        }
        for item in evidence
    ]


def validate_task_plans(
    client: ChatClient,
    tasks: list[dict],
    all_evidence: list[dict],
    stage_orders: dict[str, int],
    cache_dir: Path,
    prompt_dir: Path,
    batch_id: str,
) -> dict[str, dict]:
    template = load_template(prompt_dir, "task_plan_validation.yaml")
    available_ids = {
        stage: [
            item["evidence_id"] for item in all_evidence
            if stage_orders[item["introduced_stage"]] <= order
        ]
        for stage, order in stage_orders.items()
    }
    payload = {
        "stage_order": stage_orders,
        "evidence_catalog": evidence_payload(all_evidence),
        "available_evidence_ids_by_stage": available_ids,
        "candidates": [
            {
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "ask_after_stage": task["ask_after_stage"],
                "gold_answer": task["gold_answer"],
                "selected_evidence": task["selected_evidence"],
            }
            for task in tasks
        ],
    }
    result = cached_completion(
        client,
        template.substitute(plan_json=json.dumps(payload, ensure_ascii=False, indent=2)),
        cache_dir / "task_plan_validation_batch" / f"{batch_id}.json",
        max_tokens=8000,
    )
    reviews = {item["task_id"]: item for item in result["results"]}
    missing = [task["task_id"] for task in tasks if task["task_id"] not in reviews]
    if missing:
        raise ValueError(f"Task-plan validation omitted candidates: {missing}")
    return {
        task_id: {
            "accepted": bool(review["accepted"]),
            "repairable": bool(review.get("repairable", False)),
            "feedback": str(review.get("feedback", "")),
            "issues": review.get("issues", []) or [],
            "fixed_required_evidence_ids": review.get("fixed_required_evidence_ids"),
            "fixed_gold_answer": review.get("fixed_gold_answer"),
        }
        for task_id, review in reviews.items()
    }


def validate_task(
    client: ChatClient,
    task: dict,
    available_evidence: list[dict],
    cache_dir: Path,
    prompt_dir: Path,
    attempt: int = 1,
) -> dict:
    # 同时基于所选证据和提问时点前的完整证据校验任务。
    cache_path = cache_dir / "qa_validation" / f"{task['task_id']}_a{attempt}.json"
    template = load_template(prompt_dir, "qa_validation.yaml")
    task_for_prompt = {
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "ask_after_stage": task["ask_after_stage"],
        "question": task["question"],
        "gold_answer": task["gold_answer"],
        "required_evidence_ids": [item["evidence_id"] for item in task["selected_evidence"]],
        "selected_evidence": task["selected_evidence"],
        "all_available_evidence": evidence_payload(available_evidence),
    }
    prompt = template.substitute(task_json=json.dumps(task_for_prompt, ensure_ascii=False, indent=2))
    result = cached_completion(client, prompt, cache_path, max_tokens=8000)
    return {
        "accepted": result["accepted"],
        "feedback": str(result.get("feedback", "")),
        "issues": result.get("issues", []) or [],
        "fixed_question": result.get("fixed_question"),
        "fixed_answer": result.get("fixed_answer"),
    }


def finalize_task(
    client: ChatClient,
    spec: dict,
    available_evidence: list[dict],
    cache_dir: Path,
    verifier_client: ChatClient,
    log_prefix: str,
    prompt_dir: Path = PROMPT_DIR,
    max_iters: int = 3,
) -> dict:
    # 生成<->校验反馈循环: 若未通过校验, 把反馈交回模型重生成问题, 并应用校验给出的修正问题/修正金标准, 直到通过或达到最大轮数
    task = dict(spec)
    feedback: dict | None = None
    validation: dict = {}
    current_gold = spec["gold_answer"]
    for it in range(1, max_iters + 1):
        # (1) 生成问题(可带上一轮反馈, 使用当前(可能已修复的)金标准)
        candidate = generate_question(
            client,
            spec,
            cache_dir,
            prompt_dir,
            feedback=feedback,
            attempt=it,
            gold_answer=current_gold,
        )
        task["question"] = candidate["question"]
        task["gold_answer"] = current_gold

        # (2) 使用所选证据与完整可用历史验证问题和答案
        validation = validate_task(
            verifier_client,
            task,
            available_evidence,
            cache_dir,
            prompt_dir,
            attempt=it,
        )
        log(
            f"{log_prefix}[step3/validation] task={spec['task_id']} "
            f"attempt={it}/{max_iters} accepted={validation.get('accepted')}"
        )
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
    return task


def select_evaluation_evidence(
    client: ChatClient,
    task_id: str,
    question: str,
    answer: str,
    index: EvidenceIndex,
    cache_dir: Path,
    prompt_dir: Path = PROMPT_DIR,
) -> list[str]:
    # 只从提问时点已经释放的 EvidenceIndex 中选择权威答案所需事实。
    cache_path = cache_dir / "evaluation_evidence" / f"{task_id}.json"
    template = load_template(prompt_dir, "evidence_selection.yaml")
    prompt = template.substitute(
        question=question,
        answer=answer,
        evidence_text=evidence_catalog(index),
        edges_text=edges_text(index),
    )
    result = cached_completion(client, prompt, cache_path, max_tokens=12000)
    return result["required_evidence_ids"]


def generate_rubric(
    client: ChatClient,
    task: dict,
    cache_dir: Path,
    prompt_dir: Path,
) -> dict:
    template = load_template(prompt_dir, "rubric_generation.yaml")
    feedback = ""
    for attempt in range(1, 3):
        prompt = template.substitute(
            task_type=task["task_type"],
            question=task["question"],
            answer=task["gold_answer"],
            feedback_block=feedback,
        )
        cache_path = cache_dir / "rubric_generation" / f"{task['task_id']}_a{attempt}.json"
        result = cached_completion(client, prompt, cache_path, max_tokens=12000)
        raw_criteria = result["criteria"]
        total = sum(Decimal(str(item["score"])) for item in raw_criteria)
        if Decimal(str(result["max_score"])) == 100 and total == 100:
            criteria = [
                {
                    "name": str(item["name"]).strip(),
                    "score": float(item["score"]),
                    "description": str(item.get("description", "")),
                }
                for item in raw_criteria
            ]
            return {
                "rubric_id": f"{task['task_type']}_{task['task_id']}",
                "task_id": task["task_id"],
                "max_score": 100,
                "criteria": criteria,
            }
        feedback = (
            f"Your previous rubric declared max_score={result['max_score']} and its criteria summed to {total}, "
            "but both must equal exactly 100. Regenerate the complete rubric and reallocate the criterion scores."
        )
    raise ValueError(f"Rubric scores for {task['task_id']} do not sum to 100 after 2 attempts")
