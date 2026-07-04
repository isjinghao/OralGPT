"""Step5 评分: base 任务 ACC / ERS 与 rubric 清单打分 (TPS / 诊断)。"""
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


def judge_base(llm: CachedLLM, record: dict) -> dict:
    # 判定一条 base 任务作答是否正确，并统计 selected_evidence 中被正确覆盖的证据数(ERS)
    selected_evidence = _selected_evidence_payload(record)
    prompt = render(
        "judge_base",
        question=record["question"],
        gold=record["gold_answer"],
        answer=record["model_answer"],
        selected_evidence=json.dumps(selected_evidence, ensure_ascii=False),
    )
    data = llm.complete(prompt, cache_key=f"judge_base_{record['task_id']}", max_tokens=3000)

    evidence_items = data.get("evidence", []) or []
    total_evidence = len(selected_evidence)
    raw_covered = data.get("covered_evidence_count", None)
    if raw_covered is None:
        raw_covered = sum(1 for item in evidence_items if item.get("covered"))
    try:
        covered_evidence = int(raw_covered or 0)
    except (TypeError, ValueError):
        covered_evidence = 0
    covered_evidence = max(0, min(total_evidence, covered_evidence))

    return {
        "correct": bool(data.get("correct")),
        "reason": str(data.get("reason", "")),
        "covered_evidence_count": covered_evidence,
        "total_evidence_count": total_evidence,
        "evidence": evidence_items,
    }


def judge_rubric(llm: CachedLLM, record: dict, rubric: dict) -> dict:
    # 按 rubric 清单为一条治疗/诊断作答打分
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
    data = llm.complete(prompt, cache_key=f"rubric_{record['task_id']}", max_tokens=6000)

    # 先按名字收集模型给分
    awarded_by_name: dict[str, float] = {}
    for item in data.get("criteria", []) or []:
        awarded_by_name[str(item.get("name", "")).strip()] = float(item.get("awarded", 0) or 0)

    detailed = []
    total_awarded = 0.0
    graded = data.get("criteria", []) or []

    # 按原始 rubric 顺序逐项处理，本地代码不直接信大模型的总分
    for idx, c in enumerate(criteria_defs):
        max_score = float(c["score"])
        name = c["name"]
        # 优先按 name 对齐
        if name.strip() in awarded_by_name:
            raw = awarded_by_name[name.strip()]
        # 如果名字没对上，就按顺序兜底
        elif idx < len(graded):
            try:
                raw = float(graded[idx].get("awarded", 0) or 0)
            except (TypeError, ValueError):
                raw = 0.0
        else:
            raw = 0.0
        # 封顶
        awarded = max(0.0, min(max_score, raw))
        total_awarded += awarded
        detailed.append({"name": name, "max": max_score, "awarded": awarded})

    max_total = float(sum(c["score"] for c in criteria_defs)) or float(rubric.get("max_score", 0)) or 1.0
    return {
        "awarded": round(total_awarded, 2),
        "max_total": round(max_total, 2),
        "percent": round(total_awarded / max_total * 100, 2),
        "criteria": detailed,
    }
