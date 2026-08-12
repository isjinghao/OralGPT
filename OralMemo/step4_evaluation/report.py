"""Step5 报告: 汇总 ACC / ERS、治疗与随访评分，并对比不同记忆方法。"""
from __future__ import annotations

from batch_utils import log
from step4_evaluation.evaluator import CachedLLM
from step4_evaluation.scoring import judge_base, judge_evidence, judge_rubric

# base 任务(走 ACC 二元判定，并统计 selected_evidence 覆盖数)
BASE_TYPES = {
    "modality_perception",
    "longitudinal_evidence_recall",
    "cross_modal_reasoning",
    "memory_update_conflict_correction",
}


def task_modalities(record: dict) -> set[str]:
    # 从任务的 selected_evidence 中收集其涉及的所有模态代码
    mods: set[str] = set()
    for ev in record.get("selected_evidence", []) or []:
        for m in ev.get("modality", []) or []:
            if m:
                mods.add(m)
    return mods


def ratio(correct: int | float, total: int | float) -> float:
    return round(correct / total * 100, 2) if total else 0.0


def score_method(
    method_name: str,
    records: list[dict],
    rubric_by_task: dict[str, dict],
    llm: CachedLLM,
    log_prefix: str,
) -> dict:
    # 对单个记忆方法的所有作答记录打分并聚合
    acc_overall = {"correct": 0, "total": 0}
    ers_overall = {"covered": 0, "total": 0}

    # 按照任务类型/模态统计 ACC 与 ERS
    acc_by_type: dict[str, dict[str, int]] = {}
    acc_by_modality: dict[str, dict[str, int]] = {}
    ers_by_type: dict[str, dict[str, int]] = {}
    ers_by_modality: dict[str, dict[str, int]] = {}
    treatment: list[dict] = []
    followup: list[dict] = []
    per_task: list[dict] = []

    for task_index, rec in enumerate(records, start=1):
        ttype = rec["task_type"]
        log(
            f"{log_prefix}[step4/scoring] method={method_name} "
            f"task={task_index}/{len(records)} id={rec['task_id']}"
        )

        # base 任务
        if ttype in BASE_TYPES:
            verdict = judge_base(llm, rec)
            correct = 1 if verdict["correct"] else 0
            covered_evidence = int(verdict.get("covered_evidence_count", 0) or 0)
            total_evidence = int(verdict.get("total_evidence_count", 0) or 0)

            acc_overall["total"] += 1
            acc_overall["correct"] += correct
            ers_overall["covered"] += covered_evidence
            ers_overall["total"] += total_evidence

            acc_bt = acc_by_type.setdefault(ttype, {"correct": 0, "total": 0})
            acc_bt["total"] += 1
            acc_bt["correct"] += correct
            ers_bt = ers_by_type.setdefault(ttype, {"covered": 0, "total": 0})
            ers_bt["covered"] += covered_evidence
            ers_bt["total"] += total_evidence

            modalities = sorted(task_modalities(rec))
            for mod in modalities:
                acc_bm = acc_by_modality.setdefault(mod, {"correct": 0, "total": 0})
                acc_bm["total"] += 1
                acc_bm["correct"] += correct

            covered_ids = {
                str(item.get("evidence_id", "")).strip()
                for item in verdict.get("evidence", []) or []
                if item.get("covered")
            }
            for ev in rec.get("selected_evidence", []) or []:
                eid = str(ev.get("evidence_id", "")).strip()
                ev_covered = 1 if eid in covered_ids else 0
                for mod in ev.get("modality", []) or []:
                    ers_bm = ers_by_modality.setdefault(mod, {"covered": 0, "total": 0})
                    ers_bm["covered"] += ev_covered
                    ers_bm["total"] += 1

            per_task.append({
                "task_id": rec["task_id"],
                "task_type": ttype,
                "metric": "ACC/ERS",
                "correct": bool(correct),
                "reason": verdict["reason"],
                "covered_evidence_count": covered_evidence,
                "total_evidence_count": total_evidence,
                "ers_score": ratio(covered_evidence, total_evidence),
                "evidence": verdict.get("evidence", []),
                "modalities": modalities,
            })

        # 论文明确支持的治疗/随访决策任务
        elif ttype in ("treatment", "followup"):
            scored = judge_rubric(llm, rec, rubric_by_task[rec["task_id"]])
            entry = {"task_id": rec["task_id"], **scored}
            if ttype == "followup":
                followup.append(entry)
                metric = "followup"
            else:
                treatment.append(entry)
                metric = "TPS"

            # 证据召回(ERS): 治疗/随访任务同样有 selected_evidence
            verdict = judge_evidence(llm, rec)
            covered_evidence = int(verdict.get("covered_evidence_count", 0) or 0)
            total_evidence = int(verdict.get("total_evidence_count", 0) or 0)

            ers_overall["covered"] += covered_evidence
            ers_overall["total"] += total_evidence
            ers_bt = ers_by_type.setdefault(ttype, {"covered": 0, "total": 0})
            ers_bt["covered"] += covered_evidence
            ers_bt["total"] += total_evidence

            covered_ids = {
                str(item.get("evidence_id", "")).strip()
                for item in verdict.get("evidence", []) or []
                if item.get("covered")
            }
            for ev in rec.get("selected_evidence", []) or []:
                eid = str(ev.get("evidence_id", "")).strip()
                ev_covered = 1 if eid in covered_ids else 0
                for mod in ev.get("modality", []) or []:
                    ers_bm = ers_by_modality.setdefault(mod, {"covered": 0, "total": 0})
                    ers_bm["covered"] += ev_covered
                    ers_bm["total"] += 1

            per_task.append({
                "task_id": rec["task_id"],
                "task_type": ttype,
                "metric": metric,
                "awarded": scored["awarded"],
                "max_total": scored["max_total"],
                "percent": scored["percent"],
                "covered_evidence_count": covered_evidence,
                "total_evidence_count": total_evidence,
                "ers_score": ratio(covered_evidence, total_evidence),
                "evidence": verdict.get("evidence", []),
            })

    # 分别计算治疗和随访推理任务平均分
    tps_percent = round(sum(t["percent"] for t in treatment) / len(treatment), 2) if treatment else None
    followup_percent = round(sum(t["percent"] for t in followup) / len(followup), 2) if followup else None

    return {
        "method": method_name,
        "acc": {
            "overall": {**acc_overall, "score": ratio(acc_overall["correct"], acc_overall["total"])},
            "by_task_type": {
                k: {**v, "score": ratio(v["correct"], v["total"])}
                for k, v in sorted(acc_by_type.items())
            },
            "by_modality": {
                k: {**v, "score": ratio(v["correct"], v["total"])}
                for k, v in sorted(acc_by_modality.items())
            },
        },
        "ers": {
            "overall": {**ers_overall, "score": ratio(ers_overall["covered"], ers_overall["total"])},
            "by_task_type": {
                k: {**v, "score": ratio(v["covered"], v["total"])}
                for k, v in sorted(ers_by_type.items())
            },
            "by_modality": {
                k: {**v, "score": ratio(v["covered"], v["total"])}
                for k, v in sorted(ers_by_modality.items())
            },
        },
        "tps": {"overall_percent": tps_percent, "per_task": treatment},
        "followup": {"overall_percent": followup_percent, "per_task": followup},
        "per_task": per_task,
    }


def build_report(
    records_by_method: dict[str, list[dict]],
    rubric_by_task: dict[str, dict],
    llm_by_method: dict[str, CachedLLM],
    log_prefix: str = "[evaluation][unknown]",
) -> dict:
    """对每个记忆方法评分，汇总为总报告。"""
    methods = [
        score_method(name, recs, rubric_by_task, llm_by_method[name], log_prefix)
        for name, recs in records_by_method.items()
    ]
    return {"methods": methods}


def _fmt_pct(v) -> str:
    return f"{v:6.2f}%" if isinstance(v, (int, float)) else "   n/a"


def format_console(report: dict) -> str:
    # 将报告渲染为便于阅读的对比表格文本
    methods = report["methods"]
    names = [m["method"] for m in methods]
    col = 16
    lines: list[str] = []

    def row(label: str, values: list[str]) -> str:
        return f"{label:<34}" + "".join(f"{v:>{col}}" for v in values)

    lines.append("=" * (34 + col * len(names)))
    lines.append(row("Metric \\ Method", names))
    lines.append("-" * (34 + col * len(names)))

    # 总体 ACC
    lines.append(row("ACC overall", [
        f"{m['acc']['overall']['score']:.1f}% ({m['acc']['overall']['correct']}/{m['acc']['overall']['total']})"
        for m in methods
    ]))

    # 分任务类型 ACC
    all_acc_types = sorted({t for m in methods for t in m["acc"]["by_task_type"]})
    for t in all_acc_types:
        vals = []
        for m in methods:
            cell = m["acc"]["by_task_type"].get(t)
            vals.append(f"{cell['score']:.1f}% ({cell['correct']}/{cell['total']})" if cell else "-")
        lines.append(row(f"  ACC[{t}]", vals))

    # 分模态 ACC
    all_acc_mods = sorted({m0 for m in methods for m0 in m["acc"]["by_modality"]})
    for mod in all_acc_mods:
        vals = []
        for m in methods:
            cell = m["acc"]["by_modality"].get(mod)
            vals.append(f"{cell['score']:.1f}% ({cell['correct']}/{cell['total']})" if cell else "-")
        lines.append(row(f"  ACC-modality[{mod}]", vals))

    lines.append("-" * (34 + col * len(names)))

    # 总体 ERS(selected_evidence 覆盖率)
    lines.append(row("ERS overall", [
        f"{m['ers']['overall']['score']:.1f}% ({m['ers']['overall']['covered']}/{m['ers']['overall']['total']})"
        for m in methods
    ]))

    # 分任务类型 ERS
    all_ers_types = sorted({t for m in methods for t in m["ers"]["by_task_type"]})
    for t in all_ers_types:
        vals = []
        for m in methods:
            cell = m["ers"]["by_task_type"].get(t)
            vals.append(f"{cell['score']:.1f}% ({cell['covered']}/{cell['total']})" if cell else "-")
        lines.append(row(f"  ERS[{t}]", vals))

    # 分模态 ERS
    all_ers_mods = sorted({m0 for m in methods for m0 in m["ers"]["by_modality"]})
    for mod in all_ers_mods:
        vals = []
        for m in methods:
            cell = m["ers"]["by_modality"].get(mod)
            vals.append(f"{cell['score']:.1f}% ({cell['covered']}/{cell['total']})" if cell else "-")
        lines.append(row(f"  ERS-modality[{mod}]", vals))

    lines.append("-" * (34 + col * len(names)))

    # TPS 与随访推理
    lines.append(row("Treatment score", [
        _fmt_pct(m.get("tps", {}).get("overall_percent")) for m in methods]))
    lines.append(row("Follow-up score", [
        _fmt_pct(m.get("followup", {}).get("overall_percent")) for m in methods]))
    lines.append("=" * (34 + col * len(names)))
    return "\n".join(lines)
