from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from batch_utils import add_batch_arguments, log, patient_output_root, run_patient_batch, selected_patients
from config import get_settings
from llm_client import ChatClient
from step3_tasks.llm_tasks import (
    PROMPT_DIR,
    finalize_task,
    generate_rubric,
    load_normal_task_plan,
    plan_task_candidates,
    select_evaluation_evidence,
)
from step3_tasks.selectors import (
    EvidenceIndex,
    assemble_evaluation_task,
    assemble_normal_task,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _review_feedback(validation: dict) -> str:
    parts = [str(validation.get("feedback") or "").strip()]
    for issue in validation.get("issues", []) or []:
        parts.append(str(issue.get("problem", issue) if isinstance(issue, dict) else issue).strip())
    return "; ".join(part for part in parts if part) or "Candidate was rejected by the reviewer"


def _preflight_task_plan(
    task: dict,
    available_evidence: list[dict],
    stage_orders: dict[str, int],
    constraints: dict,
) -> str | None:
    selected = task["selected_evidence"]
    if not selected:
        return "no evidence selected"

    available_ids = {item["evidence_id"] for item in available_evidence}
    unavailable = [item["evidence_id"] for item in selected if item["evidence_id"] not in available_ids]
    if unavailable:
        return f"unavailable evidence: {unavailable}"
    if not str(task.get("gold_answer", "")).strip():
        return "gold_answer is empty"

    stages = {item["stage"] for item in selected}
    exact_stages = constraints.get("exact_stages")
    min_stages = constraints.get("min_stages")
    if exact_stages is not None and len(stages) != exact_stages:
        return f"needs exactly {exact_stages} stage(s), got {len(stages)}"
    if min_stages is not None and len(stages) < min_stages:
        return f"needs at least {min_stages} stages, got {len(stages)}"
    if constraints.get("requires_prior_evidence") and not any(
        stage_orders[stage] < stage_orders[task["ask_after_stage"]] for stage in stages
    ):
        return "needs evidence earlier than ask_after_stage"
    return None


def build_normal_tasks(
    client: ChatClient,
    patient_stages: dict,
    index: EvidenceIndex,
    cache_dir: Path,
    verifier_client: ChatClient,
    prompt_dir: Path,
    max_planning_iters: int = 3,
) -> list[dict]:
    patient_id = patient_stages["patient_id"]
    prefix = f"[benchmark][{patient_id}]"
    log(f"{prefix}[step3/planning] started")
    stage_orders = {stage["stage_id"]: int(stage["order"]) for stage in patient_stages["stages"]}
    tasks: list[dict] = []

    for entry in load_normal_task_plan(prompt_dir):
        task_type = entry["task_type"]
        target = int(entry["count"])
        accepted: list[dict] = []
        feedback: list[str] = []
        for attempt in range(1, max_planning_iters + 1):
            missing = target - len(accepted)
            if not missing:
                break
            candidates = plan_task_candidates(
                client, patient_stages, index, cache_dir, prompt_dir,
                entry, missing, attempt, feedback,
            )
            current_feedback: list[str] = []
            for candidate_no, item in enumerate(candidates[:missing], start=1):
                if item.get("task_type") != task_type:
                    current_feedback.append(f"wrong task_type: {item.get('task_type')}")
                    continue
                if item.get("ask_after_stage") not in stage_orders:
                    current_feedback.append(f"unknown stage: {item.get('ask_after_stage')}")
                    continue
                candidate_id = f"{task_type}_a{attempt}_{candidate_no:03d}"
                try:
                    spec = assemble_normal_task(patient_id, candidate_id, item, index)
                except ValueError as exc:
                    current_feedback.append(str(exc))
                    continue

                available = index.available_at(spec["ask_after_stage"], stage_orders)
                problem = _preflight_task_plan(spec, available, stage_orders, entry["constraints"])
                if problem:
                    current_feedback.append(problem)
                    continue
                related = index.related_evidence(
                    [evidence["evidence_id"] for evidence in spec["selected_evidence"]],
                    available,
                )
                task = finalize_task(
                    client,
                    spec,
                    related,
                    cache_dir,
                    verifier_client=verifier_client,
                    log_prefix=prefix,
                    prompt_dir=prompt_dir,
                )
                if task["validation"].get("accepted"):
                    task["task_id"] = f"{task_type}_{len(accepted) + 1:03d}"
                    accepted.append(task)
                else:
                    current_feedback.append(_review_feedback(task["validation"]))
            feedback = list(dict.fromkeys(current_feedback))

        if len(accepted) != target:
            raise RuntimeError(f"{task_type}: generated {len(accepted)}/{target} tasks")
        tasks.extend(accepted)

    log(f"{prefix}[step3/planning] completed accepted={len(tasks)}")
    return tasks


def build_evaluation_tasks(
    client: ChatClient,
    standard: dict,
    index: EvidenceIndex,
    cache_dir: Path,
    prompt_dir: Path,
    task_workers: int = 4,
) -> list[dict]:
    patient_id = standard["patient_id"]
    prefix = f"[benchmark][{patient_id}]"
    task_prefix = patient_id.replace("__", "_")
    stage_orders = {stage["stage_id"]: int(stage["order"]) for stage in standard["stages"]}
    counters: dict[str, int] = {}
    jobs = []
    for stage in standard["stages"]:
        task_type = stage["stage_type"]
        for turn in stage["qa_pairs"]:
            if turn["role"] != "evaluation":
                continue
            ask_after_stage = turn["ask_after_stage"]
            available = index.available_at(ask_after_stage, stage_orders)
            available_ids = {item["evidence_id"] for item in available}
            available_graph = {
                "edges": [
                    edge for edge in index.graph["edges"]
                    if edge["source"] in available_ids and edge["target"] in available_ids
                ]
            }
            counters[task_type] = counters.get(task_type, 0) + 1
            task_id = f"{task_prefix}_{task_type}_{counters[task_type]:03d}"
            jobs.append((task_id, task_type, turn, EvidenceIndex(available, available_graph)))

    def build(job: tuple[str, str, dict, EvidenceIndex]) -> dict:
        task_id, task_type, turn, available_index = job
        evidence_ids = select_evaluation_evidence(
            client, task_id, turn["human"], turn["assistant"], available_index, cache_dir, prompt_dir
        )
        task = assemble_evaluation_task(
            patient_id=patient_id,
            task_id=task_id,
            task_type=task_type,
            turn=turn,
            evidence_ids=evidence_ids,
            index=available_index,
        )
        log(
            f"{prefix}[step3/evaluation-task] task={task_id} type={task_type} "
            f"ask={turn['ask_after_stage']} evidence={len(task['selected_evidence'])}"
        )
        return task

    with ThreadPoolExecutor(max_workers=task_workers) as executor:
        return list(executor.map(build, jobs))


def build_client(settings, role: str, patient_id: str) -> ChatClient:
    cfg = settings.llm_for(role)
    return ChatClient(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=cfg.model,
        log_prefix=f"[benchmark][{patient_id}]",
    )


def run_patient(
    out: Path,
    patient_id: str,
    settings,
    prompt_dir: Path = PROMPT_DIR,
    task_workers: int = 4,
) -> None:
    prefix = f"[benchmark][{patient_id}]"
    standard = read_json(out / "trajectories" / "standard_trajectory.json")
    evidence_data = read_json(out / "evidence" / "evidence.json")
    evidence_graph = read_json(out / "graph" / "evidence_graph.json")
    index = EvidenceIndex(evidence=evidence_data["evidence"], graph=evidence_graph)
    cache_dir = out / "cache" / "step3"
    client = build_client(settings, "benchmark", patient_id)
    verifier_client = build_client(settings, "verifier", patient_id)

    log(f"{prefix}[step3/start] evidence={len(evidence_data['evidence'])}")
    tasks = build_normal_tasks(
        client,
        standard,
        index,
        cache_dir,
        verifier_client,
        prompt_dir,
    )
    tasks.extend(build_evaluation_tasks(client, standard, index, cache_dir, prompt_dir, task_workers))

    rubric_tasks = [task for task in tasks if task["task_type"] in {"treatment", "followup"}]

    def build_rubric(item: tuple[int, dict]) -> dict:
        index_number, task = item
        log(f"{prefix}[step3/rubric] task={index_number}/{len(rubric_tasks)} id={task['task_id']}")
        return generate_rubric(client, task, cache_dir, prompt_dir)

    with ThreadPoolExecutor(max_workers=task_workers) as executor:
        treatment_rubrics = list(executor.map(build_rubric, enumerate(rubric_tasks, start=1)))

    groups: dict[str, list[dict]] = {}
    for task in tasks:
        groups.setdefault(task["task_type"], []).append(task)
    tasks_dir = out / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    for path in tasks_dir.glob("*.json"):
        path.unlink()
    write_json(tasks_dir / "all_tasks.json", {"patient_id": patient_id, "tasks": tasks})
    for group_name, items in groups.items():
        write_json(tasks_dir / f"{group_name}.json", {"patient_id": patient_id, "tasks": items})
    write_json(out / "rubrics" / "treatment_rubrics.json", treatment_rubrics)
    log(f"{prefix}[step3/done] tasks={len(tasks)} rubrics={len(treatment_rubrics)} groups={len(groups)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark tasks and rubrics")
    add_batch_arguments(parser)
    parser.add_argument("--task-workers", type=int, default=4)
    return parser.parse_args()


def completed(out: Path) -> bool:
    return (out / "tasks" / "all_tasks.json").is_file() and (
        out / "rubrics" / "treatment_rubrics.json"
    ).is_file()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    patients = selected_patients(settings.dataset_json, args.all, args.limit)

    def worker(item: dict) -> str:
        patient_id = item["id"]
        out = patient_output_root(settings.bench_root, patient_id)
        if not args.force and completed(out):
            log(f"[benchmark][{patient_id}][step3/resume] completed outputs found; skipped")
            return "skipped"
        run_patient(out, patient_id, settings, task_workers=args.task_workers)
        return "completed"

    return run_patient_batch(patients, args.num_workers, "benchmark", worker)


if __name__ == "__main__":
    raise SystemExit(main())
