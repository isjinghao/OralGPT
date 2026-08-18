from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from string import Template

import yaml

from utils.batch_utils import log
from llm_client import ChatClient
from step3_tasks.selectors import (
    EvidenceIndex,
    edges_text,
    evidence_catalog,
    evidence_ref,
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


def cached_completion(
    client: ChatClient,
    prompt: str,
    cache_path: Path,
    max_tokens: int,
    valid: Callable[[dict], bool] | None = None,
) -> dict:
    cache_input = {"model": client.model, "prompt": prompt, "max_tokens": max_tokens}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        result = cached.get("result")
        if cached.get("input") == cache_input and isinstance(result, dict) and (valid is None or valid(result)):
            client.log("step3/cache", f"cache_hit file={cache_path.name}")
            return result
    result = client.complete_json(prompt, max_tokens=max_tokens)
    if valid is None or valid(result):
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
    def valid(value: dict) -> bool:
        return isinstance(value.get("question"), str) and bool(value["question"].strip())

    result = cached_completion(client, prompt, cache_path, max_tokens=8000, valid=valid)
    if valid(result):
        return result

    repair_prompt = (
        f"{prompt}\n\nYour previous JSON omitted a non-empty 'question'. "
        "Return only {\"question\": \"...\"}.\n\nPrevious JSON:\n"
        f"{json.dumps(result, ensure_ascii=False)}"
    )
    result = cached_completion(
        client,
        repair_prompt,
        cache_path.with_stem(f"{cache_path.stem}_repair"),
        max_tokens=2000,
        valid=valid,
    )
    if not valid(result):
        raise ValueError("question_generation returned no non-empty 'question' after schema repair")
    return result


def evidence_payload(evidence: list[dict]) -> list[dict]:
    return [evidence_ref(item) for item in evidence]


def validate_task(
    client: ChatClient,
    task: dict,
    related_evidence: list[dict],
    cache_dir: Path,
    prompt_dir: Path,
    attempt: int = 1,
) -> dict:
    cache_path = cache_dir / "qa_validation" / f"{task['task_id']}_a{attempt}.json"
    template = load_template(prompt_dir, "qa_validation.yaml")
    task_for_prompt = {
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "ask_after_stage": task["ask_after_stage"],
        "question": task["question"],
        "gold_answer": task["gold_answer"],
        "selected_evidence": task["selected_evidence"],
        "related_evidence": evidence_payload(related_evidence),
    }
    prompt = template.substitute(task_json=json.dumps(task_for_prompt, ensure_ascii=False, indent=2))
    result = cached_completion(client, prompt, cache_path, max_tokens=2000)
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
    related_evidence: list[dict],
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
        question = candidate.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question_generation returned no non-empty 'question'")
        task["question"] = question
        task["gold_answer"] = current_gold

        # (2) 使用所选证据与直接相关历史验证问题和答案
        validation = validate_task(
            verifier_client,
            task,
            related_evidence,
            cache_dir,
            prompt_dir,
            attempt=it,
        )
        log(
            f"{log_prefix}[step3/validation] task={spec['task_id']} "
            f"attempt={it}/{max_iters} accepted={validation.get('accepted')}"
        )
        if validation.get("fixed_answer"):
            current_gold = validation["fixed_answer"]
        if validation.get("fixed_question"):
            task["question"] = validation["fixed_question"]
        if validation.get("accepted"):
            break

        # (3) 未通过: 把反馈交回下一轮重新生成/复验
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
    def valid(result: dict) -> bool:
        evidence_ids = result.get("required_evidence_ids")
        try:
            return isinstance(evidence_ids, list) and bool(evidence_ids) and bool(index.resolve(evidence_ids))
        except (TypeError, ValueError):
            return False

    result = cached_completion(client, prompt, cache_path, max_tokens=12000, valid=valid)
    evidence_ids = result.get("required_evidence_ids")
    try:
        if not valid(result):
            raise ValueError("required_evidence_ids must be a non-empty list from the catalog")
        return list(dict.fromkeys(evidence_ids))
    except (TypeError, ValueError) as exc:
        repair_prompt = (
            f"{prompt}\n\nThe previous JSON was invalid: {exc}. "
            "Return only exact ids copied from the catalog in "
            "{\"required_evidence_ids\": [\"...\"]}.\n\nPrevious JSON:\n"
            f"{json.dumps(result, ensure_ascii=False)}"
        )
        repaired = cached_completion(
            client,
            repair_prompt,
            cache_path.with_stem(f"{cache_path.stem}_repair"),
            max_tokens=2000,
            valid=valid,
        )
        if not valid(repaired):
            raise ValueError("evidence selection repair returned invalid evidence IDs")
        return list(dict.fromkeys(repaired["required_evidence_ids"]))


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
