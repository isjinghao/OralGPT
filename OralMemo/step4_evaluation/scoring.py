"""Step5 评分: base 任务 ACC / ERS 与 rubric 清单打分。"""
from __future__ import annotations

import json

from step4_evaluation.evaluator import CachedLLM
from step4_evaluation.templating import render


def _selected_evidence_payload(record: dict) -> list[dict]:
    return [
        {
            "evidence_id": ev.get("evidence_id", ""),
            "stage": ev.get("stage", ""),
            "modality": ev.get("modality", []),
            "fact_text": ev.get("fact_text", ""),
            "field": ev.get("field", ""),
            "value": ev.get("value"),
            "unit": ev.get("unit"),
        }
        for ev in record.get("selected_evidence", []) or []
    ]


def _normalize_coverage(selected_evidence: list[dict], judged_items: list[dict], covered_ids: set[str] | None = None) -> tuple[list[dict], int, int]:
    judged_by_id = {
        str(item.get("evidence_id", "")).strip(): item
        for item in judged_items
        if str(item.get("evidence_id", "")).strip()
    }
    normalized = []
    covered_count = 0
    for evidence in selected_evidence:
        evidence_id = str(evidence.get("evidence_id", "")).strip()
        judged = judged_by_id.get(evidence_id, {})
        covered = (
            evidence_id in covered_ids
            if covered_ids is not None
            else bool(judged.get("covered", False))
        )
        item = {
            "evidence_id": evidence_id,
            "covered": covered,
        }
        if judged.get("reason"):
            item["reason"] = str(judged["reason"])
        normalized.append(item)
        covered_count += int(covered)
    return normalized, covered_count, len(normalized)


def judge_base(llm: CachedLLM, record: dict) -> dict:
    # 判定 base 任务，并对预先筛选的全部 selected_evidence 统一计算 ERS
    selected_evidence = _selected_evidence_payload(record)
    prompt = render(
        "judge_base",
        question=record["question"],
        gold=record["gold_answer"],
        answer=record["model_answer"],
        selected_evidence=json.dumps(selected_evidence, ensure_ascii=False),
    )
    data = llm.complete(prompt, cache_key=f"judge_base_{record['task_id']}", max_tokens=8000)
    evidence, covered_evidence, total_evidence = _normalize_coverage(
        selected_evidence, data.get("evidence", []) or []
    )
    return {
        "correct": bool(data.get("correct")),
        "reason": str(data.get("reason", "")),
        "covered_evidence_count": covered_evidence,
        "total_evidence_count": total_evidence,
        "evidence": evidence,
    }


def judge_evidence(llm: CachedLLM, record: dict) -> dict:
    # 仅测量证据召回；selected_evidence 已在 benchmark 生成阶段筛为该题所需证据
    selected_evidence = _selected_evidence_payload(record)
    compact = [
        {
            "evidence_id": ev.get("evidence_id", ""),
            "fact_text": ev.get("fact_text", ""),
            "modality": ev.get("modality", []),
        }
        for ev in selected_evidence
    ]
    prompt = render(
        "judge_evidence",
        question=record["question"],
        gold=record["gold_answer"],
        answer=record["model_answer"],
        selected_evidence=json.dumps(compact, ensure_ascii=False),
    )
    data = llm.complete(prompt, cache_key=f"judge_evidence_{record['task_id']}", max_tokens=16000)
    evidence, covered_evidence, total_evidence = _normalize_coverage(
        selected_evidence,
        [],
        {str(x).strip() for x in (data.get("covered_evidence_ids", []) or [])},
    )
    return {
        "covered_evidence_count": covered_evidence,
        "total_evidence_count": total_evidence,
        "evidence": evidence,
    }


def judge_rubric(llm: CachedLLM, record: dict, rubric: dict) -> dict:
    # 按 rubric 清单为一条治疗/随访作答打分
    criteria_defs = rubric.get("criteria", [])
    payload = [
        {
            "name": c["name"],
            "max_score": c["score"],
            "description": c.get("description", "")
        }
        for c in criteria_defs
    ]
    prompt = render(
        "judge_rubric",
        question=record["question"],
        gold=record["gold_answer"],
        answer=record["model_answer"],
        criteria=json.dumps(payload, ensure_ascii=False),
    )
    data = llm.complete(prompt, cache_key=f"rubric_{record['task_id']}", max_tokens=16000)

    graded = data["criteria"]
    graded_by_name = {str(item["name"]).strip(): item for item in graded}
    expected_names = {str(item["name"]).strip() for item in criteria_defs}
    if len(graded_by_name) != len(graded) or set(graded_by_name) != expected_names:
        raise ValueError(f"Rubric criteria mismatch for task {record['task_id']}")

    detailed = []
    total_awarded = 0.0
    for criterion in criteria_defs:
        max_score = float(criterion["score"])
        name = str(criterion["name"]).strip()
        graded_item = graded_by_name[name]
        raw = float(graded_item["awarded"])
        awarded = max(0.0, min(max_score, raw))
        total_awarded += awarded
        detailed.append({
            "name": name,
            "max": max_score,
            "awarded": awarded,
            "reason": str(graded_item.get("reason", "")),
        })

    max_total = float(sum(criterion["score"] for criterion in criteria_defs))
    if max_total <= 0:
        raise ValueError(f"Rubric max score must be positive: {record['task_id']}")
    return {
        "awarded": round(total_awarded, 2),
        "max_total": round(max_total, 2),
        "percent": round(total_awarded / max_total * 100, 2),
        "criteria": detailed,
    }
