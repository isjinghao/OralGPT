from __future__ import annotations

import argparse
import json
from pathlib import Path

from batch_utils import add_batch_arguments, log, patient_output_root, run_patient_batch, selected_patients
from config import get_settings
from llm_client import ChatClient
from step4_evaluation.evaluator import CachedLLM, run_streaming
from step4_evaluation.memory import available_methods, build_methods
from step4_evaluation.report import build_report, format_console


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def group_tasks_by_stage(tasks: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for task in tasks:
        grouped.setdefault(task["ask_after_stage"], []).append(task)
    return grouped


def load_rubric_index(out: Path) -> dict[str, dict]:
    rubrics = read_json(out / "rubrics" / "treatment_rubrics.json")
    return {rubric["task_id"]: rubric for rubric in rubrics}


def resolve_trajectory_path(out: Path, name: str) -> Path:
    if name == "standard":
        return out / "trajectories" / "standard_trajectory.json"
    if name == "model_perception":
        return out / "trajectories" / "model_perception_trajectory.json"
    return out / "variants" / f"{name}.json"


def build_client(settings, role: str, patient_id: str) -> ChatClient:
    cfg = settings.llm_for(role)
    return ChatClient(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=cfg.model,
        log_prefix=f"[evaluation][{patient_id}]",
    )


def evaluate_trajectory(
    trajectory: dict,
    tasks_by_stage: dict[str, list[dict]],
    rubric_by_task: dict[str, dict],
    answer_client: ChatClient,
    verifier_client: ChatClient,
    out: Path,
    methods: list[str],
    multimodal: bool,
    image_root: Path | None,
) -> dict:
    patient_id = trajectory["patient_id"]
    prefix = f"[evaluation][{patient_id}]"
    trajectory_type = trajectory["trajectory_type"]
    mode = "multimodal" if multimodal else "text"
    suffix = "_mm" if multimodal else ""
    present_stages = {stage["stage_id"] for stage in trajectory["stages"]}
    missing = sorted(stage_id for stage_id in tasks_by_stage if stage_id not in present_stages)
    log(
        f"{prefix}[step4/trajectory] started type={trajectory_type} mode={mode} "
        f"stages={len(trajectory['stages'])} missing_task_stages={len(missing)}"
    )

    eval_dir = out / "evaluation" / f"{trajectory_type}{suffix}"
    cache_root = out / "cache" / "step4" / f"{trajectory_type}{suffix}"
    records_by_method: dict[str, list[dict]] = {}
    llm_by_method: dict[str, CachedLLM] = {}
    for method in build_methods(names=methods, multimodal=multimodal):
        log(f"{prefix}[step4/method] started name={method.name}")
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
            log_prefix=prefix,
        )
        records_by_method[method.name] = records
        write_json(eval_dir / f"answers_{method.name}.json", records)
        log(
            f"{prefix}[step4/method] completed name={method.name} answers={len(records)} "
            f"llm_calls={answer_llm.calls} cache_hits={answer_llm.hits}"
        )

    log(f"{prefix}[step4/scoring] started trajectory={trajectory_type}")
    report = build_report(records_by_method, rubric_by_task, llm_by_method, log_prefix=prefix)
    report.update(
        {
            "trajectory_id": trajectory["trajectory_id"],
            "trajectory_type": trajectory_type,
            "mode": mode,
            "patient_id": patient_id,
        }
    )
    write_json(eval_dir / "report.json", report)
    (eval_dir / "report.txt").write_text(format_console(report), encoding="utf-8")
    for method_report in report["methods"]:
        log(
            f"{prefix}[step4/result] trajectory={trajectory_type} method={method_report['method']} "
            f"acc={method_report['acc']['overall']['score']:.2f} "
            f"ers={method_report['ers']['overall']['score']:.2f} "
            f"treatment={method_report['tps']['overall_percent']}"
        )
    log(f"{prefix}[step4/trajectory] completed type={trajectory_type} mode={mode}")
    return report


def trajectory_completed(
    out: Path,
    trajectory_name: str,
    methods: list[str],
    multimodal: bool,
) -> bool:
    trajectory_path = resolve_trajectory_path(out, trajectory_name)
    if not trajectory_path.is_file():
        return False
    trajectory_type = read_json(trajectory_path)["trajectory_type"]
    suffix = "_mm" if multimodal else ""
    eval_dir = out / "evaluation" / f"{trajectory_type}{suffix}"
    report_path = eval_dir / "report.json"
    if not report_path.is_file() or not (eval_dir / "report.txt").is_file():
        return False
    report_methods = {item["method"] for item in read_json(report_path).get("methods", [])}
    return set(methods) <= report_methods and all(
        (eval_dir / f"answers_{method}.json").is_file() for method in methods
    )


def run_patient(
    out: Path,
    patient_id: str,
    settings,
    trajectory_names: list[str],
    methods: list[str],
    multimodal: bool,
) -> None:
    tasks = read_json(out / "tasks" / "all_tasks.json")["tasks"]
    tasks_by_stage = group_tasks_by_stage(tasks)
    rubric_by_task = load_rubric_index(out)
    answer_client = build_client(settings, "answer", patient_id)
    verifier_client = build_client(settings, "verifier", patient_id)
    image_root = settings.bench_root if multimodal else None
    log(
        f"[evaluation][{patient_id}][step4/start] trajectories={','.join(trajectory_names)} "
        f"methods={','.join(methods)} multimodal={multimodal} tasks={len(tasks)}"
    )
    for trajectory_name in trajectory_names:
        trajectory_path = resolve_trajectory_path(out, trajectory_name)
        if not trajectory_path.is_file():
            raise FileNotFoundError(f"Required trajectory does not exist: {trajectory_path}")
        evaluate_trajectory(
            read_json(trajectory_path),
            tasks_by_stage,
            rubric_by_task,
            answer_client,
            verifier_client,
            out,
            methods,
            multimodal,
            image_root,
        )
    log(f"[evaluation][{patient_id}][step4/done] trajectories={len(trajectory_names)}")


def parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated benchmarks")
    add_batch_arguments(parser)
    parser.add_argument("--trajectories", type=parse_csv, default=["standard"])
    parser.add_argument("--methods", type=parse_csv, default=["full_context_memory"])
    parser.add_argument("--multimodal", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    unknown_methods = sorted(set(args.methods) - set(available_methods()))
    if unknown_methods:
        raise ValueError(f"Unknown memory methods: {unknown_methods}")
    settings = get_settings()
    patients = selected_patients(settings.dataset_json, args.all, args.limit)

    def worker(item: dict) -> str:
        patient_id = item["id"]
        out = patient_output_root(settings.bench_root, patient_id)
        if not args.force and all(
            trajectory_completed(out, name, args.methods, args.multimodal)
            for name in args.trajectories
        ):
            log(f"[evaluation][{patient_id}][step4/resume] completed outputs found; skipped")
            return "skipped"
        run_patient(out, patient_id, settings, args.trajectories, args.methods, args.multimodal)
        return "completed"

    return run_patient_batch(patients, args.num_workers, "evaluation", worker)


if __name__ == "__main__":
    raise SystemExit(main())
