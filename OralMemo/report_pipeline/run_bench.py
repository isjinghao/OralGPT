"""在 report_pipeline 产物上跑 step2(证据/图) + step3(任务/rubric) + step4(评测)。

用法:
    python report_pipeline/run_bench.py --name pls_8y                 # 跑 step2,3,4
    python report_pipeline/run_bench.py --name pls_8y --steps 2,3     # 只跑 step2,3
    python report_pipeline/run_bench.py --name pls_8y --methods single_stage_memory,summary_memory
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

if "bench" not in sys.modules:
    _bench = types.ModuleType("bench")
    _bench.__path__ = [str(BENCH_ROOT)]
    sys.modules["bench"] = _bench

from bench.config import get_settings, load_env
from bench.llm_client import ChatClient
from bench.step2_evidence.evidence import extract_all_evidence
from bench.step2_evidence.graph import build_evidence_graph
from bench.step3_tasks.llm_tasks import generate_rubric, select_heldout_evidence
from bench.step3_tasks.run_step3_chenfang import build_normal_tasks
from bench.step3_tasks.selectors import EvidenceIndex, assemble_heldout_task

from step4_evaluation.evaluator import CachedLLM, run_streaming
from step4_evaluation.memory import build_methods
from step4_evaluation.report import build_report, format_console


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_client(settings, role: str = "benchmark") -> ChatClient:
    cfg = settings.llm_for(role)
    return ChatClient(api_key=cfg.api_key, base_url=cfg.base_url, model=cfg.model)


# ------------------------------ step2 ------------------------------
def run_step2(out: Path, client: ChatClient, settings) -> None:
    standard = read_json(out / "trajectories" / "standard_trajectory.json")
    evidence = extract_all_evidence(client, standard, cache_dir=out / "cache")
    write_json(out / "evidence" / "evidence.json", evidence)
    graph = build_evidence_graph(
        out / "evidence" / "evidence.json",
        client=client,
        cache_dir=out / "cache",
        max_edges=settings.graph_max_edges,
    )
    write_json(out / "graph" / "evidence_graph.json", graph)
    print(f"[step2] 证据={len(evidence['evidence'])} 边={len(graph['edges'])}", flush=True)


# ------------------------------ step3 ------------------------------
def build_paper_tasks(client, standard, index, cache_dir):
    """从标准轨迹中读取任意数量的 treatment/followup evaluation QA。"""
    patient_id = standard["patient_id"]
    prefix = patient_id.replace("__", "_")
    stage_orders = {stage["stage_id"]: stage["order"] for stage in standard["stages"]}
    counters: dict[str, int] = {}
    tasks = []
    evaluation_turns = [
        {**turn, "source_stage_id": stage["stage_id"]}
        for stage in standard["stages"]
        for turn in stage["qa_pairs"]
        if turn["role"] == "evaluation"
    ]
    for turn in evaluation_turns:
        evaluation_type = turn["evaluation_type"]
        task_type = f"paper_{evaluation_type}"
        ask_after_stage = turn["ask_after_stage"]
        available = index.available_at(ask_after_stage, stage_orders)
        available_ids = {item["evidence_id"] for item in available}
        available_graph = {
            **index.graph,
            "edges": [
                edge for edge in index.graph["edges"]
                if edge["source"] in available_ids and edge["target"] in available_ids
            ],
        }
        available_index = EvidenceIndex(evidence=available, graph=available_graph)
        counters[task_type] = counters.get(task_type, 0) + 1
        task_id = f"{prefix}_{task_type}_{counters[task_type]:03d}"
        evidence_ids = select_heldout_evidence(
            client,
            task_id,
            turn["human"],
            turn["assistant"],
            available_index,
            cache_dir,
        )
        task = assemble_heldout_task(
            patient_id=patient_id,
            task_id=task_id,
            task_type=task_type,
            ask_after_stage=ask_after_stage,
            turn=turn,
            evidence_ids=evidence_ids,
            index=available_index,
        )
        task.update(
            {
                "release_to_memory": True,
                "release_after_stage": turn["release_after_stage"],
                "release_group": evaluation_type,
                "source_stage_id": turn["source_stage_id"],
            }
        )
        tasks.append(task)
        print(
            f"[step3 paper] {task_id} ({task_type}) @ {ask_after_stage} "
            f"-> release {turn['release_after_stage']} | evidence={len(task['selected_evidence'])}",
            flush=True,
        )
    return tasks


def run_step3(out: Path, client: ChatClient, verifier_client: ChatClient) -> None:
    standard = read_json(out / "trajectories" / "standard_trajectory.json")
    evidence_data = read_json(out / "evidence" / "evidence.json")
    evidence_graph = read_json(out / "graph" / "evidence_graph.json")
    index = EvidenceIndex(evidence=evidence_data["evidence"], graph=evidence_graph)
    cache_dir = out / "cache" / "step3"

    tasks = build_normal_tasks(client, standard, index, cache_dir, verifier_client=verifier_client)
    tasks.extend(build_paper_tasks(client, standard, index, cache_dir))

    rubrics = {"diagnosis_rubrics": [], "treatment_rubrics": []}
    for task in tasks:
        if task["task_type"] in {"paper_treatment", "paper_followup"}:
            rubrics["treatment_rubrics"].append(generate_rubric(client, task, cache_dir))

    groups: dict[str, list[dict]] = {}
    for task in tasks:
        groups.setdefault(task["task_type"], []).append(task)

    pid = standard["patient_id"]
    write_json(out / "tasks" / "all_tasks.json", {"patient_id": pid, "tasks": tasks})
    for name, items in groups.items():
        write_json(out / "tasks" / f"{name}.json", {"patient_id": pid, "tasks": items})
    write_json(out / "rubrics" / "diagnosis_rubrics.json", rubrics["diagnosis_rubrics"])
    write_json(out / "rubrics" / "treatment_rubrics.json", rubrics["treatment_rubrics"])
    print(f"[step3] 任务={len(tasks)} 分组={ {k: len(v) for k, v in groups.items()} }", flush=True)


# ------------------------------ step4 ------------------------------
def group_tasks_by_stage(tasks: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for task in tasks:
        grouped.setdefault(task["ask_after_stage"], []).append(task)
    return grouped


def load_rubric_index(out: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for name in ("diagnosis_rubrics.json", "treatment_rubrics.json"):
        path = out / "rubrics" / name
        if path.exists():
            for rubric in read_json(path):
                index[rubric["task_id"]] = rubric
    return index


def run_step4(
    out: Path,
    answer_client: ChatClient,
    verifier_client: ChatClient,
    methods: list[str] | None,
    multimodal: bool,
    release_perception_ground_truth: bool,
    release_treatment_ground_truth: bool,
) -> None:
    trajectory_name = (
        "standard_trajectory.json"
        if release_perception_ground_truth
        else "model_perception_trajectory.json"
    )
    trajectory_path = out / "trajectories" / trajectory_name
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Required trajectory does not exist: {trajectory_path}")
    trajectory = read_json(trajectory_path)
    tasks = read_json(out / "tasks" / "all_tasks.json")["tasks"]
    tasks_by_stage = group_tasks_by_stage(tasks)
    rubric_by_task = load_rubric_index(out)
    image_root = BENCH_ROOT if multimodal else None

    perception_tag = "perception_gt" if release_perception_ground_truth else "perception_model"
    treatment_tag = "treatment_gt" if release_treatment_ground_truth else "treatment_model"
    multimodal_tag = "_mm" if multimodal else ""
    experiment_name = f"{perception_tag}__{treatment_tag}{multimodal_tag}"
    eval_dir = out / "evaluation" / experiment_name
    cache_root = out / "cache" / "step4" / experiment_name

    records_by_method: dict[str, list[dict]] = {}
    llm_by_method: dict[str, CachedLLM] = {}
    for method in build_methods(names=methods, multimodal=multimodal):
        method_dir = cache_root / method.name
        method.setup(method_dir)
        answer_llm = CachedLLM(answer_client, method_dir / "answer")
        verifier_llm = CachedLLM(verifier_client, method_dir / "verifier")
        llm_by_method[method.name] = verifier_llm
        records = run_streaming(
            method,
            trajectory,
            tasks_by_stage,
            answer_llm,
            image_root,
            release_treatment_ground_truth=release_treatment_ground_truth,
        )
        records_by_method[method.name] = records
        write_json(eval_dir / f"answers_{method.name}.json", records)
        print(f"[step4] method={method.name} 作答={len(records)}", flush=True)

    report = build_report(records_by_method, rubric_by_task, llm_by_method)
    report.update({
        "trajectory_id": trajectory["trajectory_id"],
        "trajectory_type": trajectory["trajectory_type"],
        "patient_id": trajectory["patient_id"],
        "release_perception_ground_truth": release_perception_ground_truth,
        "release_treatment_ground_truth": release_treatment_ground_truth,
    })
    write_json(eval_dir / "report.json", report)
    console = format_console(report)
    (eval_dir / "report.txt").write_text(console, encoding="utf-8")
    print("\n" + console)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=lambda s: [x.strip() for x in s.split(",")], default=["2", "3", "4"])
    parser.add_argument("--methods", type=lambda s: s.split(","), default=None)
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument(
        "--release-perception-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use paper-ground-truth image observations instead of the model-perception trajectory",
    )
    parser.add_argument(
        "--release-treatment-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Release paper treatment answers, rather than model treatment answers, before follow-up",
    )
    args = parser.parse_args()

    load_env(BENCH_ROOT / ".env")
    settings = get_settings()
    benchmark_client = build_client(settings, "benchmark")
    answer_client = build_client(settings, "answer")
    verifier_client = build_client(settings, "verifier")
    out = BENCH_ROOT / "outputs" / "report" / args.name

    if "2" in args.steps:
        run_step2(out, benchmark_client, settings)
    if "3" in args.steps:
        run_step3(out, benchmark_client, verifier_client)
    if "4" in args.steps:
        run_step4(
            out,
            answer_client,
            verifier_client,
            args.methods,
            args.multimodal,
            args.release_perception_ground_truth,
            args.release_treatment_ground_truth,
        )


if __name__ == "__main__":
    main()
