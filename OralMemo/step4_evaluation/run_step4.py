from __future__ import annotations

import argparse
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Semaphore

from config import get_settings
from utils.batch_utils import add_batch_arguments, log, patient_output_root, run_patient_batch, selected_patients
from utils.json_utils import read_json, write_json
from llm_client import ChatClient, build_client
from step4_evaluation.evaluator import CachedLLM, run_streaming
from step4_evaluation.memory import available_methods, build_methods
from step4_evaluation.report import format_csv, score_method


def group_tasks_by_stage(tasks: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for task in tasks:
        grouped.setdefault(task["ask_after_stage"], []).append(task)
    return grouped


def load_rubric_index(out: Path) -> dict[str, dict]:
    rubrics = read_json(out / "rubrics" / "treatment_rubrics.json")
    return {rubric["task_id"]: rubric for rubric in rubrics}


def resolve_trajectory_path(out: Path, name: str, answer_model: str | None = None) -> Path:
    if name == "standard_trajectory":
        return out / "trajectories" / "standard_trajectory.json"
    if name == "model_perception_trajectory":
        if not answer_model:
            raise ValueError("answer model is required for model_perception_trajectory")
        return out / "trajectories" / name / answer_model / f"{name}.json"
    return out / "trajectories" / name / f"{name}.json"


def evaluation_roots(out: Path, trajectory_type: str, answer_model: str) -> tuple[Path, Path]:
    """Return result and cache roots for one answer model."""
    base = out / "evaluation" / trajectory_type / answer_model
    cache = out / "cache" / "step4" / trajectory_type / answer_model
    return base, cache


def _score_and_checkpoint(
    method_name: str,
    records: list[dict],
    memory_metrics: dict,
    rubric_by_task: dict[str, dict],
    verifier_llm: CachedLLM,
    prefix: str,
    report_path: Path,
    score_workers: int,
    score_semaphore: Semaphore,
) -> dict:
    method_report = score_method(
        method_name,
        records,
        rubric_by_task,
        verifier_llm,
        prefix,
        score_workers,
        score_semaphore,
    )
    method_report["memory_metrics"] = memory_metrics
    write_json(report_path, method_report)
    return method_report


def evaluate_trajectory(
    trajectory: dict,
    tasks_by_stage: dict[str, list[dict]],
    rubric_by_task: dict[str, dict],
    answer_client: ChatClient,
    verifier_client: ChatClient,
    memo_client: ChatClient,
    out: Path,
    methods: list[str],
    *,
    force: bool = False,
    answer_workers: int = 2,
    score_workers: int = 1,
    method_workers: int = 1,
) -> dict:
    patient_id = trajectory["patient_id"]
    prefix = f"[evaluation][{patient_id}]"
    trajectory_type = trajectory["trajectory_type"]
    present_stages = {stage["stage_id"] for stage in trajectory["stages"]}
    missing = sorted(stage_id for stage_id in tasks_by_stage if stage_id not in present_stages)
    log(
        f"{prefix}[step4/trajectory] started type={trajectory_type} "
        f"stages={len(trajectory['stages'])} missing_task_stages={len(missing)}"
    )

    eval_root, cache_root = evaluation_roots(out, trajectory_type, answer_client.model)
    answer_semaphore = Semaphore(answer_workers)
    score_semaphore = Semaphore(score_workers)
    method_reports: dict[str, dict] = {}
    method_objects = build_methods(names=methods)

    def answer_method(method):
        method_dir = cache_root / method.name
        method_eval_dir = eval_root / method.name
        answers_path = method_eval_dir / "answers.json"
        metrics_path = method_eval_dir / "memory_metrics.json"
        method_report_path = method_eval_dir / "report.json"
        cached_records = read_json(answers_path) if answers_path.is_file() and not force else None
        answers_rebuilt = cached_records is None or not metrics_path.is_file()
        if not answers_rebuilt:
            records = cached_records
            memory_metrics = read_json(metrics_path)
            log(f"{prefix}[step4/resume] method={method.name} answers=loaded")
        else:
            log(f"{prefix}[step4/method] started name={method.name}")
            if force and method_dir.exists():
                shutil.rmtree(method_dir)
            method_memo_client = ChatClient(
                memo_client.api_key,
                memo_client.base_url,
                memo_client.model,
                memo_client.log_prefix,
            )
            if metrics_path.is_file() and not force:
                method.restore_metrics(read_json(metrics_path))
            namespace = f"{patient_id}:{trajectory['trajectory_id']}:{method.name}"
            setup_succeeded = False
            try:
                method.setup(method_dir, namespace)
                setup_succeeded = True
                records = run_streaming(
                    method,
                    trajectory,
                    tasks_by_stage,
                    CachedLLM(answer_client, method_dir / "answer"),
                    answer_semaphore=answer_semaphore,
                    answer_workers=answer_workers,
                    memory_llm=CachedLLM(method_memo_client, method_dir / "memo"),
                    log_prefix=prefix,
                )
                memory_metrics = method.metrics()
                write_json(answers_path, records)
            except Exception:
                if not setup_succeeded:
                    method.add_metrics(failures=1)
                raise
            finally:
                write_json(metrics_path, method.metrics())
                try:
                    method.close()
                finally:
                    method_memo_client.close()
            log(f"{prefix}[step4/method] completed name={method.name} answers={len(records)}")
        return method, method_dir, method_report_path, records, memory_metrics, answers_rebuilt

    def score_answered(answered) -> dict:
        method, method_dir, report_path, records, memory_metrics, answers_rebuilt = answered
        if report_path.is_file() and not force and not answers_rebuilt:
            cached_report = read_json(report_path)
            if not cached_report.get("failed_tasks") and "memory_metrics" in cached_report:
                log(f"{prefix}[step4/resume] method={method.name} scoring=loaded")
                return cached_report
            log(f"{prefix}[step4/resume] method={method.name} retry_failed_scoring")
        return _score_and_checkpoint(
            method.name,
            records,
            memory_metrics,
            rubric_by_task,
            CachedLLM(verifier_client, method_dir / "verifier"),
            prefix,
            report_path,
            score_workers,
            score_semaphore,
        )

    if method_workers > 1:
        def evaluate_method(method) -> dict:
            return score_answered(answer_method(method))

        with ThreadPoolExecutor(max_workers=method_workers) as executor:
            reports = executor.map(evaluate_method, method_objects)
            method_reports = {method.name: report for method, report in zip(method_objects, reports)}
    else:
        score_futures: dict[str, Future] = {}
        use_pipeline = answer_client.base_url != verifier_client.base_url
        score_executor = ThreadPoolExecutor(max_workers=1) if use_pipeline else None
        try:
            for method in method_objects:
                answered = answer_method(method)
                if score_executor is None:
                    method_reports[method.name] = score_answered(answered)
                else:
                    score_futures[method.name] = score_executor.submit(score_answered, answered)
            for method_name, future in score_futures.items():
                method_reports[method_name] = future.result()
        finally:
            if score_executor is not None:
                score_executor.shutdown(wait=True)

    report = {
        "methods": [method_reports[name] for name in methods],
        "trajectory_id": trajectory["trajectory_id"],
        "trajectory_type": trajectory_type,
        "patient_id": patient_id,
        "answer_model": answer_client.model,
        "answer_base_url": answer_client.base_url,
        "verifier_model": verifier_client.model,
        "verifier_base_url": verifier_client.base_url,
        "memo_model": memo_client.model,
        "memo_base_url": memo_client.base_url,
        "memory_methods": methods,
    }
    write_json(eval_root / "report.json", report)
    (eval_root / "report.csv").write_text(format_csv(report), encoding="utf-8")
    (eval_root / "report.txt").unlink(missing_ok=True)
    for method_report in report["methods"]:
        log(
            f"{prefix}[step4/result] trajectory={trajectory_type} method={method_report['method']} "
            f"acc={method_report['acc']['overall']['score']:.2f} "
            f"ers={method_report['ers']['overall']['score']:.2f} "
            f"treatment={method_report['tps']['overall_percent']} "
            f"failed_tasks={len(method_report.get('failed_tasks', []))}"
        )
    log(f"{prefix}[step4/trajectory] completed type={trajectory_type}")
    return report


def trajectory_completed(
    out: Path,
    trajectory_name: str,
    methods: list[str],
    answer_model: str,
) -> bool:
    trajectory_path = resolve_trajectory_path(out, trajectory_name, answer_model)
    if not trajectory_path.is_file():
        return False
    trajectory_type = read_json(trajectory_path)["trajectory_type"]
    eval_root, _ = evaluation_roots(out, trajectory_type, answer_model)
    report_path = eval_root / "report.json"
    if not report_path.is_file() or not (eval_root / "report.csv").is_file():
        return False
    method_reports = read_json(report_path).get("methods", [])
    report_methods = {item["method"] for item in method_reports}
    if any(item.get("failed_tasks") or "memory_metrics" not in item for item in method_reports):
        return False
    return set(methods) <= report_methods and all(
        (eval_root / method / "answers.json").is_file()
        and (eval_root / method / "report.json").is_file()
        for method in methods
    )


def run_patient(
    out: Path,
    patient_id: str,
    settings,
    trajectory_names: list[str],
    methods: list[str],
    *,
    answer_model: str,
    answer_base_url: str | None = None,
    force: bool = False,
    answer_workers: int = 2,
    score_workers: int = 1,
    method_workers: int = 1,
) -> None:
    tasks = read_json(out / "tasks" / "all_tasks.json")["tasks"]
    tasks_by_stage = group_tasks_by_stage(tasks)
    rubric_by_task = load_rubric_index(out)
    with (
        build_client(
            settings,
            "answer",
            patient_id,
            log_prefix="[evaluation]",
            model=answer_model,
            base_url=answer_base_url,
        ) as answer_client,
        build_client(settings, "verifier", patient_id, log_prefix="[evaluation]") as verifier_client,
        build_client(settings, "memo", patient_id, log_prefix="[evaluation]") as memo_client,
    ):
        log(
            f"[evaluation][{patient_id}][step4/start] trajectories={','.join(trajectory_names)} "
            f"methods={','.join(methods)} tasks={len(tasks)}"
        )
        failed_tasks = 0
        for trajectory_name in trajectory_names:
            trajectory_path = resolve_trajectory_path(out, trajectory_name, answer_client.model)
            if not trajectory_path.is_file():
                raise FileNotFoundError(f"Required trajectory does not exist: {trajectory_path}")
            report = evaluate_trajectory(
                read_json(trajectory_path),
                tasks_by_stage,
                rubric_by_task,
                answer_client,
                verifier_client,
                memo_client,
                out,
                methods,
                force=force,
                answer_workers=answer_workers,
                score_workers=score_workers,
                method_workers=method_workers,
            )
            failed_tasks += sum(len(item.get("failed_tasks", [])) for item in report["methods"])
        if failed_tasks:
            raise RuntimeError(f"{failed_tasks} scoring tasks failed; rerun the same command to retry them")
        log(f"[evaluation][{patient_id}][step4/done] trajectories={len(trajectory_names)}")


def parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated benchmarks")
    add_batch_arguments(parser)
    parser.add_argument("--trajectories", type=parse_csv, default=["standard_trajectory"])
    parser.add_argument("--methods", type=parse_csv, default=["full_context_memory"])
    parser.add_argument("--answer-model", default=None, help="Override ANSWER_OPENAI_MODEL for this run")
    parser.add_argument("--answer-base-url", default=None, help="Override ANSWER_OPENAI_BASE_URL for this run")
    parser.add_argument("--answer-workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--score-workers", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--method-workers", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    unknown_methods = sorted(set(args.methods) - set(available_methods()))
    if unknown_methods:
        raise ValueError(f"Unknown memory methods: {unknown_methods}")
    settings = get_settings()
    patients = selected_patients(settings.dataset_json, args.all, args.limit)
    answer_model = args.answer_model or settings.llm_for("answer").model

    def worker(item: dict) -> str:
        patient_id = item["id"]
        out = patient_output_root(settings.bench_root, patient_id)
        if not args.force and all(
            trajectory_completed(
                out,
                name,
                args.methods,
                answer_model,
            )
            for name in args.trajectories
        ):
            log(f"[evaluation][{patient_id}][step4/resume] completed outputs found; skipped")
            return "skipped"
        run_patient(
            out,
            patient_id,
            settings,
            args.trajectories,
            args.methods,
            answer_model=answer_model,
            answer_base_url=args.answer_base_url,
            force=args.force,
            answer_workers=args.answer_workers,
            score_workers=args.score_workers,
            method_workers=args.method_workers,
        )
        return "completed"

    return run_patient_batch(patients, args.num_workers, "evaluation", worker)


if __name__ == "__main__":
    raise SystemExit(main())
