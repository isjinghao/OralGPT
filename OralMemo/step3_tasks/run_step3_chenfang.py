from __future__ import annotations

import json
from pathlib import Path

from bench.config import get_settings
from bench.llm_client import ChatClient
from bench.step3_tasks.llm_tasks import (
    finalize_task,
    generate_rubric,
    plan_normal_tasks,
    select_heldout_evidence,
)
from bench.step3_tasks.selectors import EvidenceIndex, assemble_heldout_task, assemble_normal_task


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_normal_tasks(client: ChatClient, patient_stages: dict, index: EvidenceIndex, cache_dir: Path) -> list[dict]:
    # 构建普通任务
    planned = plan_normal_tasks(client, patient_stages, index, cache_dir)
    patient_id = patient_stages["patient_id"]
    counters: dict[str, int] = {}
    tasks = []
    for item in planned:
        task_type = item["task_type"]
        counters[task_type] = counters.get(task_type, 0) + 1
        spec = assemble_normal_task(patient_id, f"{task_type}_{counters[task_type]:03d}", item, index)
        print(f"[Step3 normal] {spec['task_id']} ({spec['task_type']})", flush=True)
        task = finalize_task(client, spec, cache_dir)
        print(f"  accepted={task['validation'].get('accepted')}", flush=True)
        tasks.append(task)
    return tasks


def build_heldout_tasks(client: ChatClient, patient_stages: dict, index: EvidenceIndex, cache_dir: Path) -> list[dict]:
    # 构建 held-out 诊断/治疗任务
    patient_id = patient_stages["patient_id"]
    prefix = patient_id.replace("__", "_")
    counters: dict[str, int] = {}
    tasks = []
    for turn in patient_stages["heldout_turns"]:
        task_type = turn["role"]
        counters[task_type] = counters.get(task_type, 0) + 1
        task_id = f"{prefix}_{task_type}_{counters[task_type]:03d}"
        evidence_ids = select_heldout_evidence(client, task_id, turn["human"], turn["assistant"], index, cache_dir)
        task = assemble_heldout_task(
            patient_id=patient_id,
            task_id=task_id,
            task_type=task_type,
            ask_after_stage="S5_TMJ",
            turn=turn,
            evidence_ids=evidence_ids,
            index=index,
        )
        print(
            f"[Step3 heldout] {task_id} ({task_type}) <- turn {turn['source_turn_id']} "
            f"| evidence={len(task['selected_evidence'])}",
            flush=True,
        )
        tasks.append(task)
    return tasks


def main() -> None:
    # (1) 读取 Step1/Step2 产物
    settings = get_settings()
    out = settings.output_root
    patient_stages = read_json(out / "stages" / "patient_stages.json")
    evidence_data = read_json(out / "evidence" / "evidence.json")
    evidence_graph = read_json(out / "graph" / "evidence_graph.json")

    client = ChatClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
    )
    index = EvidenceIndex(evidence=evidence_data["evidence"], graph=evidence_graph)
    cache_dir = out / "cache" / "step3"

    # (2) LLM 规划并生成普通任务
    tasks = build_normal_tasks(client, patient_stages, index, cache_dir)

    # (3) 拆分 held-out 诊断/治疗任务并由 LLM 归因证据
    tasks.extend(build_heldout_tasks(client, patient_stages, index, cache_dir))

    # (4) 为诊断/治疗任务生成 rubric
    rubrics = {"diagnosis_rubrics": [], "treatment_rubrics": []}
    for task in tasks:
        if task["task_type"] == "heldout_diagnosis":
            rubrics["diagnosis_rubrics"].append(generate_rubric(client, task, cache_dir))
        elif task["task_type"] == "heldout_treatment":
            rubrics["treatment_rubrics"].append(generate_rubric(client, task, cache_dir))

    # (5) 写 all_tasks.json、各分组 json 与 rubric 文件
    groups: dict[str, list[dict]] = {}
    for task in tasks:
        groups.setdefault(task["task_type"], []).append(task)

    tasks_dir = out / "tasks"
    write_json(tasks_dir / "all_tasks.json", {"patient_id": patient_stages["patient_id"], "tasks": tasks})
    for group_name, items in groups.items():
        write_json(tasks_dir / f"{group_name}.json", {"patient_id": patient_stages["patient_id"], "tasks": items})

    rubric_dir = out / "rubrics"
    write_json(rubric_dir / "diagnosis_rubrics.json", rubrics["diagnosis_rubrics"])
    write_json(rubric_dir / "treatment_rubrics.json", rubrics["treatment_rubrics"])

    result = {
        "patient_id": patient_stages["patient_id"],
        "task_count": len(tasks),
        "accepted_count": sum(1 for task in tasks if task.get("validation", {}).get("accepted", True)),
        "group_counts": {name: len(items) for name, items in sorted(groups.items())},
        "rubric_counts": {key: len(value) for key, value in rubrics.items()},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
