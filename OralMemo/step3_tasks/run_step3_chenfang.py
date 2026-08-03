from __future__ import annotations

import json
from pathlib import Path

from config import get_settings
from llm_client import ChatClient
from step3_tasks.llm_tasks import (
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
    verifier_client: ChatClient | None = None,
) -> list[dict]:
    # 构建普通任务; 仅保留通过校验的问题, 未通过校验的直接丢弃
    planned = plan_normal_tasks(client, patient_stages, index, cache_dir)
    patient_id = patient_stages["patient_id"]
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
        plan_validation = validate_task_plan(
            verifier_client or client,
            spec,
            available_evidence,
            cache_dir,
        )
        if not plan_validation["accepted"]:
            dropped += 1
            print(f"  dropped (plan validation failed): {spec['task_id']}", flush=True)
            continue
        print(f"[Step3 normal] {spec['task_id']} ({spec['task_type']})", flush=True)
        task = finalize_task(
            client,
            spec,
            available_evidence,
            cache_dir,
            verifier_client=verifier_client,
        )
        if not task["validation"].get("accepted"):
            dropped += 1
            print(f"  dropped (validation failed after retries): {spec['task_id']}", flush=True)
            continue
        print(f"  accepted={task['validation'].get('accepted')}", flush=True)
        tasks.append(task)
    if dropped:
        print(f"[Step3 normal] dropped {dropped} task(s) that failed validation", flush=True)
    return tasks


def build_evaluation_tasks(client: ChatClient, standard: dict, index: EvidenceIndex, cache_dir: Path) -> list[dict]:
    """从标准轨迹构造 treatment/followup evaluation 任务。"""
    patient_id = standard["patient_id"]
    prefix = patient_id.replace("__", "_")
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
            available_index = EvidenceIndex(evidence=available, graph=available_graph)
            counters[task_type] = counters.get(task_type, 0) + 1
            task_id = f"{prefix}_{task_type}_{counters[task_type]:03d}"
            evidence_ids = select_evaluation_evidence(
                client,
                task_id,
                turn["human"],
                turn["assistant"],
                available_index,
                cache_dir,
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
            print(
                f"[Step3 evaluation] {task_id} ({task_type}) @ {ask_after_stage} "
                f"-> release {turn['release_after_stage']} | evidence={len(task['selected_evidence'])}",
                flush=True,
            )
    return tasks


def main() -> None:
    # (1) 读取 Step1/Step2 产物
    settings = get_settings()
    out = settings.output_root
    standard = read_json(out / "trajectories" / "standard_trajectory.json")
    evidence_data = read_json(out / "evidence" / "evidence.json")
    evidence_graph = read_json(out / "graph" / "evidence_graph.json")

    benchmark_cfg = settings.llm_for("benchmark")
    verifier_cfg = settings.llm_for("verifier")
    client = ChatClient(
        api_key=benchmark_cfg.api_key,
        base_url=benchmark_cfg.base_url,
        model=benchmark_cfg.model,
    )
    verifier_client = ChatClient(
        api_key=verifier_cfg.api_key,
        base_url=verifier_cfg.base_url,
        model=verifier_cfg.model,
    )
    index = EvidenceIndex(evidence=evidence_data["evidence"], graph=evidence_graph)
    cache_dir = out / "cache" / "step3"

    # (2) LLM 规划并生成普通任务
    tasks = build_normal_tasks(client, standard, index, cache_dir, verifier_client=verifier_client)

    # (3) 从标准轨迹中的 evaluation QA 构造治疗/随访任务
    tasks.extend(build_evaluation_tasks(client, standard, index, cache_dir))

    # (4) 为治疗/随访任务生成 rubric
    treatment_rubrics = [
        generate_rubric(client, task, cache_dir)
        for task in tasks
        if task["task_type"] in {"treatment", "followup"}
    ]

    # (5) 写 all_tasks.json、各分组 json 与 rubric 文件
    groups: dict[str, list[dict]] = {}
    for task in tasks:
        groups.setdefault(task["task_type"], []).append(task)

    tasks_dir = out / "tasks"
    write_json(tasks_dir / "all_tasks.json", {"patient_id": standard["patient_id"], "tasks": tasks})
    for group_name, items in groups.items():
        write_json(tasks_dir / f"{group_name}.json", {"patient_id": standard["patient_id"], "tasks": items})

    rubric_dir = out / "rubrics"
    write_json(rubric_dir / "treatment_rubrics.json", treatment_rubrics)

    result = {
        "patient_id": standard["patient_id"],
        "task_count": len(tasks),
        "group_counts": {name: len(items) for name, items in sorted(groups.items())},
        "rubric_count": len(treatment_rubrics),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
