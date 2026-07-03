"""Step5 评分: 关键证据召回二元判定 (ERS) 与 rubric 清单打分 (TPS / 诊断)。"""
from __future__ import annotations

import json

from step4_evaluation.evaluator import CachedLLM
from step4_evaluation.templating import render


def judge_recall(llm: CachedLLM, record: dict) -> dict:
    """判定一条召回类作答是否正确(捕获金标准的关键事实)。

    输出: {"correct": bool, "reason": str}
    """
    prompt = render(
        "judge_recall",
        question=record["question"],
        gold=record["gold_answer"],
        answer=record["model_answer"],
    )
    data = llm.complete(prompt, cache_key=f"judge_{record['task_id']}", max_tokens=2000)
    return {"correct": bool(data.get("correct")), "reason": str(data.get("reason", ""))}


def judge_rubric(llm: CachedLLM, record: dict, rubric: dict) -> dict:
    """按 rubric 清单为一条治疗/诊断作答打分。

    输入: rubric - {max_score, criteria:[{name, score, description, ...}]}
    输出: {awarded, max_total, percent, criteria:[{name, max, awarded, reason}]}
    """
    criteria_defs = rubric.get("criteria", [])
    payload = [
        {"name": c["name"], "max_score": c["score"], "description": c.get("description", "")}
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

    # 按名称对齐评审结果与 rubric 定义, 逐项封顶到 max_score
    awarded_by_name: dict[str, float] = {}
    for item in data.get("criteria", []) or []:
        try:
            awarded_by_name[str(item.get("name", "")).strip()] = float(item.get("awarded", 0) or 0)
        except (TypeError, ValueError):
            continue

    detailed = []
    total_awarded = 0.0
    graded = data.get("criteria", []) or []
    for idx, c in enumerate(criteria_defs):
        max_score = float(c["score"])
        name = c["name"]
        if name.strip() in awarded_by_name:
            raw = awarded_by_name[name.strip()]
        elif idx < len(graded):
            try:
                raw = float(graded[idx].get("awarded", 0) or 0)
            except (TypeError, ValueError):
                raw = 0.0
        else:
            raw = 0.0
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
