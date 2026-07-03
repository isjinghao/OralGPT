"""Step5 报告: 汇总 ERS(总体/分模态/分任务类型)、诊断分与 TPS, 并对比不同记忆方法。"""
from __future__ import annotations

from step4_evaluation.evaluator import CachedLLM
from step4_evaluation.scoring import judge_recall, judge_rubric

# 召回类任务(走 ERS 二元判定)
RECALL_TYPES = {
    "modality_perception",
    "longitudinal_evidence_recall",
    "cross_modal_reasoning",
    "memory_update_conflict_correction",
}


def task_modalities(record: dict) -> set[str]:
    """从任务的 selected_evidence 中收集其涉及的所有模态代码。"""
    mods: set[str] = set()
    for ev in record.get("selected_evidence", []) or []:
        for m in ev.get("modality", []) or []:
            if m:
                mods.add(m)
    return mods


def _ratio(correct: int, total: int) -> float:
    return round(correct / total * 100, 2) if total else 0.0


def score_method(
    method_name: str,
    records: list[dict],
    rubric_by_task: dict[str, dict],
    llm: CachedLLM,
) -> dict:
    """对单个记忆方法的所有作答记录打分并聚合。"""
    ers_overall = {"correct": 0, "total": 0}
    by_type: dict[str, dict[str, int]] = {}
    by_modality: dict[str, dict[str, int]] = {}
    diagnosis = None
    treatment: list[dict] = []
    per_task: list[dict] = []

    for rec in records:
        ttype = rec["task_type"]

        if ttype in RECALL_TYPES:
            verdict = judge_recall(llm, rec)
            correct = 1 if verdict["correct"] else 0
            ers_overall["total"] += 1
            ers_overall["correct"] += correct
            bt = by_type.setdefault(ttype, {"correct": 0, "total": 0})
            bt["total"] += 1
            bt["correct"] += correct
            for mod in task_modalities(rec):
                bm = by_modality.setdefault(mod, {"correct": 0, "total": 0})
                bm["total"] += 1
                bm["correct"] += correct
            per_task.append({
                "task_id": rec["task_id"], "task_type": ttype, "metric": "ERS",
                "correct": bool(correct), "reason": verdict["reason"],
                "modalities": sorted(task_modalities(rec)),
                "validation_accepted": rec.get("validation_accepted", True),
            })

        elif ttype in ("heldout_diagnosis", "heldout_treatment"):
            rubric = rubric_by_task.get(rec["task_id"])
            if not rubric:
                continue
            scored = judge_rubric(llm, rec, rubric)
            entry = {"task_id": rec["task_id"], **scored}
            if ttype == "heldout_diagnosis":
                diagnosis = entry
                metric = "diagnosis"
            else:
                treatment.append(entry)
                metric = "TPS"
            per_task.append({
                "task_id": rec["task_id"], "task_type": ttype, "metric": metric,
                "awarded": scored["awarded"], "max_total": scored["max_total"],
                "percent": scored["percent"],
            })

    tps_percent = round(sum(t["percent"] for t in treatment) / len(treatment), 2) if treatment else None

    return {
        "method": method_name,
        "ers": {
            "overall": {**ers_overall, "score": _ratio(ers_overall["correct"], ers_overall["total"])},
            "by_task_type": {
                k: {**v, "score": _ratio(v["correct"], v["total"])} for k, v in sorted(by_type.items())
            },
            "by_modality": {
                k: {**v, "score": _ratio(v["correct"], v["total"])} for k, v in sorted(by_modality.items())
            },
        },
        "diagnosis": diagnosis,
        "tps": {"overall_percent": tps_percent, "per_task": treatment},
        "per_task": per_task,
    }


def build_report(
    records_by_method: dict[str, list[dict]],
    rubric_by_task: dict[str, dict],
    llm_by_method: dict[str, CachedLLM],
) -> dict:
    """对每个记忆方法评分, 汇总为总报告。"""
    methods = [score_method(name, recs, rubric_by_task, llm_by_method[name])
               for name, recs in records_by_method.items()]
    return {"methods": methods}


def _fmt_pct(v) -> str:
    return f"{v:6.2f}%" if isinstance(v, (int, float)) else "   n/a"


def format_console(report: dict) -> str:
    """将报告渲染为便于阅读的对比表格文本。"""
    methods = report["methods"]
    names = [m["method"] for m in methods]
    col = 16
    lines: list[str] = []

    def row(label: str, values: list[str]) -> str:
        return f"{label:<34}" + "".join(f"{v:>{col}}" for v in values)

    lines.append("=" * (34 + col * len(names)))
    lines.append(row("Metric \\ Method", names))
    lines.append("-" * (34 + col * len(names)))

    # 总体 ERS
    lines.append(row("ERS overall", [
        f"{m['ers']['overall']['score']:.1f}% ({m['ers']['overall']['correct']}/{m['ers']['overall']['total']})"
        for m in methods
    ]))

    # 分任务类型 ERS
    all_types = sorted({t for m in methods for t in m["ers"]["by_task_type"]})
    for t in all_types:
        vals = []
        for m in methods:
            cell = m["ers"]["by_task_type"].get(t)
            vals.append(f"{cell['score']:.1f}% ({cell['correct']}/{cell['total']})" if cell else "-")
        lines.append(row(f"  ERS[{t}]", vals))

    lines.append("-" * (34 + col * len(names)))
    # 分模态 ERS
    all_mods = sorted({m0 for m in methods for m0 in m["ers"]["by_modality"]})
    for mod in all_mods:
        vals = []
        for m in methods:
            cell = m["ers"]["by_modality"].get(mod)
            vals.append(f"{cell['score']:.1f}% ({cell['correct']}/{cell['total']})" if cell else "-")
        lines.append(row(f"  ERS-modality[{mod}]", vals))

    lines.append("-" * (34 + col * len(names)))
    # 诊断分
    lines.append(row("Diagnosis score", [
        _fmt_pct(m["diagnosis"]["percent"]) if m.get("diagnosis") else "   n/a" for m in methods
    ]))
    # TPS
    lines.append(row("TPS (treatment mean)", [_fmt_pct(m["tps"]["overall_percent"]) for m in methods]))
    lines.append("=" * (34 + col * len(names)))
    return "\n".join(lines)
