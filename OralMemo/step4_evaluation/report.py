"""Step5 报告: 汇总 ACC / ERS、治疗与随访评分，并对比不同记忆方法。"""
from __future__ import annotations

import csv
import io
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore

from utils.batch_utils import log
from step4_evaluation.evaluator import CachedLLM
from step4_evaluation.scoring import judge_base, judge_evidence, judge_rubric

# base 任务(走 ACC 二元判定，并统计 selected_evidence 覆盖数)
BASE_TYPES = {
    "modality_perception",
    "longitudinal_evidence_recall",
    "cross_modal_reasoning",
    "cross_temporal_reasoning",
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


def _judge_record(
    record: dict,
    rubric_by_task: dict[str, dict],
    llm: CachedLLM,
    semaphore: Semaphore,
    max_tokens: int,
) -> tuple[dict | None, dict | None]:
    with semaphore:
        if record["task_type"] in BASE_TYPES:
            return None, judge_base(llm, record, max_tokens=max_tokens)
        return (
            judge_rubric(llm, record, rubric_by_task[record["task_id"]], max_tokens=max_tokens),
            judge_evidence(llm, record, max_tokens=max_tokens),
        )


def default_judge_max_tokens(log_prefix: str) -> int:
    return 2048 if "[report__" in log_prefix else 4096


def score_method(
    method_name: str,
    records: list[dict],
    rubric_by_task: dict[str, dict],
    llm: CachedLLM,
    log_prefix: str,
    score_workers: int,
    score_semaphore: Semaphore,
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
    judge_max_tokens = default_judge_max_tokens(log_prefix)

    with ThreadPoolExecutor(max_workers=score_workers) as executor:
        futures = [
            executor.submit(_judge_record, rec, rubric_by_task, llm, score_semaphore, judge_max_tokens)
            for rec in records
        ]

    failed_tasks: list[dict] = []
    for task_index, (rec, future) in enumerate(zip(records, futures), start=1):
        ttype = rec["task_type"]
        log(
            f"{log_prefix}[step4/scoring] method={method_name} "
            f"task={task_index}/{len(records)} id={rec['task_id']}"
        )
        try:
            scored, verdict = future.result()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log(f"{log_prefix}[step4/scoring][error] task={rec['task_id']} {error}")
            failed_tasks.append({"task_id": rec["task_id"], "task_type": ttype, "error": error})
            per_task.append({
                "task_id": rec["task_id"],
                "task_type": ttype,
                "metric": "ERROR",
                "error": error,
            })
            continue

        # base 任务
        if ttype in BASE_TYPES:
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
            entry = {"task_id": rec["task_id"], **scored}
            if ttype == "followup":
                followup.append(entry)
                metric = "followup"
            else:
                treatment.append(entry)
                metric = "TPS"

            # 证据召回(ERS): 治疗/随访任务同样有 selected_evidence
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
        "failed_tasks": failed_tasks,
        "per_task": per_task,
    }



def _fmt_pct(value) -> str:
    return f"{value:.2f}%" if isinstance(value, (int, float)) else "n/a"


def format_csv(report: dict) -> str:
    """将各记忆方法的汇总统计渲染为 CSV。"""
    methods = report["methods"]
    rows: list[list[str]] = [["Metric", *(method["method"] for method in methods)]]

    rows.append([
        "ACC overall",
        *(f"{method['acc']['overall']['score']:.1f}% "
          f"({method['acc']['overall']['correct']}/{method['acc']['overall']['total']})"
          for method in methods),
    ])

    all_acc_types = sorted({task_type for method in methods for task_type in method["acc"]["by_task_type"]})
    for task_type in all_acc_types:
        values = []
        for method in methods:
            cell = method["acc"]["by_task_type"].get(task_type)
            values.append(f"{cell['score']:.1f}% ({cell['correct']}/{cell['total']})" if cell else "-")
        rows.append([f"ACC[{task_type}]", *values])

    all_acc_modalities = sorted({modality for method in methods for modality in method["acc"]["by_modality"]})
    for modality in all_acc_modalities:
        values = []
        for method in methods:
            cell = method["acc"]["by_modality"].get(modality)
            values.append(f"{cell['score']:.1f}% ({cell['correct']}/{cell['total']})" if cell else "-")
        rows.append([f"ACC-modality[{modality}]", *values])

    rows.append([
        "ERS overall",
        *(f"{method['ers']['overall']['score']:.1f}% "
          f"({method['ers']['overall']['covered']}/{method['ers']['overall']['total']})"
          for method in methods),
    ])

    all_ers_types = sorted({task_type for method in methods for task_type in method["ers"]["by_task_type"]})
    for task_type in all_ers_types:
        values = []
        for method in methods:
            cell = method["ers"]["by_task_type"].get(task_type)
            values.append(f"{cell['score']:.1f}% ({cell['covered']}/{cell['total']})" if cell else "-")
        rows.append([f"ERS[{task_type}]", *values])

    all_ers_modalities = sorted({modality for method in methods for modality in method["ers"]["by_modality"]})
    for modality in all_ers_modalities:
        values = []
        for method in methods:
            cell = method["ers"]["by_modality"].get(modality)
            values.append(f"{cell['score']:.1f}% ({cell['covered']}/{cell['total']})" if cell else "-")
        rows.append([f"ERS-modality[{modality}]", *values])

    rows.append(["Treatment score", *(_fmt_pct(method.get("tps", {}).get("overall_percent")) for method in methods)])
    rows.append(["Follow-up score", *(_fmt_pct(method.get("followup", {}).get("overall_percent")) for method in methods)])
    rows.append(["Failed scoring tasks", *(str(len(method.get("failed_tasks", []))) for method in methods)])
    for key in (
        "write_calls", "write_seconds", "write_avg_seconds", "retrieval_calls",
        "retrieval_seconds", "retrieval_avg_seconds", "llm_calls",
        "input_tokens", "output_tokens", "embedding_calls", "embedding_tokens",
        "failures", "failure_rate",
    ):
        rows.append([
            f"Memory {key}",
            *(str(method.get("memory_metrics", {}).get(key, 0)) for method in methods),
        ])

    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue()
