from __future__ import annotations

import argparse
import json
from pathlib import Path

from batch_utils import add_batch_arguments, log, patient_output_root, run_patient_batch, selected_patients
from config import get_settings
from llm_client import ChatClient
from step3_tasks.llm_tasks import (
    PROMPT_DIR,
    finalize_task,
    generate_rubric,
    plan_normal_tasks,
    select_evaluation_evidence,
    validate_task_plan,
)
from step3_tasks.selectors import EvidenceIndex, assemble_evaluation_task, assemble_normal_task


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_normal_tasks(
    client: ChatClient,
    patient_stages: dict,
    index: EvidenceIndex,
    cache_dir: Path,
    verifier_client: ChatClient,
    prompt_dir: Path,
) -> list[dict]:
    patient_id = patient_stages["patient_id"]
    prefix = f"[benchmark][{patient_id}]"
    log(f"{prefix}[step3/planning] started")
    planned = plan_normal_tasks(client, patient_stages, index, cache_dir, prompt_dir)
    stage_orders = {stage["stage_id"]: int(stage["order"]) for stage in patient_stages["stages"]}
    valid_stage_ids = set(stage_orders)
    counters: dict[str, int] = {}
    tasks = []
    dropped = 0
    for item in planned:
        task_type = item["task_type"]
        counters[task_type] = counters.get(task_type, 0) + 1
        if item["ask_after_stage"] not in valid_stage_ids:
            raise ValueError(f"Unknown ask_after_stage: {item['ask_after_stage']}")
        spec = assemble_normal_task(patient_id, f"{task_type}_{counters[task_type]:03d}", item, index)
        available_evidence = index.available_at(spec["ask_after_stage"], stage_orders)
        available_ids = {evidence["evidence_id"] for evidence in available_evidence}
        future_ids = [
            evidence["evidence_id"] for evidence in spec["selected_evidence"]
            if evidence["evidence_id"] not in available_ids
        ]
        if future_ids:
            raise ValueError(f"Evidence released after ask_after_stage: {future_ids}")
        plan_validation = validate_task_plan(verifier_client, spec, available_evidence, cache_dir)
        if not plan_validation["accepted"]:
            dropped += 1
            log(f"{prefix}[step3/planning] task={spec['task_id']} dropped=plan_validation")
            continue
        log(f"{prefix}[step3/question] task={spec['task_id']} type={spec['task_type']}")
        task = finalize_task(
            client,
            spec,
            available_evidence,
            cache_dir,
            verifier_client=verifier_client,
            log_prefix=prefix,
            prompt_dir=prompt_dir,
        )
        if not task["validation"].get("accepted"):
            dropped += 1
            log(f"{prefix}[step3/question] task={spec['task_id']} dropped=qa_validation")
            continue
        tasks.append(task)
        log(f"{prefix}[step3/question] task={spec['task_id']} accepted")
    log(f"{prefix}[step3/planning] completed accepted={len(tasks)} dropped={dropped}")
    return tasks


def build_evaluation_tasks(
    client: ChatClient,
    standard: dict,
    index: EvidenceIndex,
    cache_dir: Path,
    prompt_dir: Path,
) -> list[dict]:
    patient_id = standard["patient_id"]
    prefix = f"[benchmark][{patient_id}]"
    task_prefix = patient_id.replace("__", "_")
    stage_orders = {stage["stage_id"]: int(stage["order"]) for stage in standard["stages"]}
    counters: dict[str, int] = {}
    tasks = []
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
            available_index = EvidenceIndex(evidence=available, graph=available_graph)
            evidence_ids = select_evaluation_evidence(
                client,
                task_id,
                turn["human"],
                turn["assistant"],
                available_index,
                cache_dir,
                prompt_dir,
            )
            task = assemble_evaluation_task(
                patient_id=patient_id,
                task_id=task_id,
                task_type=task_type,
                turn=turn,
                evidence_ids=evidence_ids,
                index=available_index,
            )
            tasks.append(task)
            log(
                f"{prefix}[step3/evaluation-task] task={task_id} type={task_type} "
                f"ask={ask_after_stage} evidence={len(task['selected_evidence'])}"
            )
    return tasks


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
    tasks.extend(build_evaluation_tasks(client, standard, index, cache_dir, prompt_dir))

    rubric_tasks = [task for task in tasks if task["task_type"] in {"treatment", "followup"}]
    treatment_rubrics = []
    for index_number, task in enumerate(rubric_tasks, start=1):
        log(f"{prefix}[step3/rubric] task={index_number}/{len(rubric_tasks)} id={task['task_id']}")
        treatment_rubrics.append(generate_rubric(client, task, cache_dir))

    groups: dict[str, list[dict]] = {}
    for task in tasks:
        groups.setdefault(task["task_type"], []).append(task)
    tasks_dir = out / "tasks"
    write_json(tasks_dir / "all_tasks.json", {"patient_id": patient_id, "tasks": tasks})
    for group_name, items in groups.items():
        write_json(tasks_dir / f"{group_name}.json", {"patient_id": patient_id, "tasks": items})
    write_json(out / "rubrics" / "treatment_rubrics.json", treatment_rubrics)
    log(f"{prefix}[step3/done] tasks={len(tasks)} rubrics={len(treatment_rubrics)} groups={len(groups)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark tasks and rubrics")
    add_batch_arguments(parser)
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
        run_patient(out, patient_id, settings)
        return "completed"

    return run_patient_batch(patients, args.num_workers, "benchmark", worker)


if __name__ == "__main__":
    raise SystemExit(main())
