"""Evaluate saved model-perception trajectories for patient or report benchmarks."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import get_settings
from llm_client import ChatClient
from report_pipeline.paths import REPORT_OUTPUT_ROOT, REPORT_PDF_DIR
from step1_patient_trajectory.perception_evaluation import PerceptionEvaluator
from step2_evidence.run_perception_trajectory import DIRECT_CONTEXT_STAGE_IDS
from utils.batch_utils import add_batch_arguments, log, patient_output_root, run_patient_batch, selected_patients, selected_reports
from utils.json_utils import read_json


def case_root_for_item(dataset: str, settings, item: dict) -> Path:
    if dataset == "report":
        return REPORT_OUTPUT_ROOT / item["name"]
    return patient_output_root(settings.bench_root, item["id"])


def should_evaluate(dataset: str, stage: dict, qa: dict) -> bool:
    images = qa.get("image_paths", []) or []
    if dataset == "report":
        return qa.get("role") == "observation" and bool(images)
    return qa.get("role") != "evaluation" and bool(images) and stage["stage_id"] not in DIRECT_CONTEXT_STAGE_IDS


def model_answers_by_source(trajectory: dict) -> dict[tuple[str, int], str]:
    answers: dict[tuple[str, int], str] = {}
    for stage in trajectory.get("stages", []):
        stage_id = stage["stage_id"]
        for qa in stage.get("qa_pairs", []):
            key = (stage_id, int(qa.get("source_turn_id", 0)))
            answers[key] = (qa.get("assistant") or "").strip()
    return answers


def evaluate_case(dataset: str, item: dict, settings, model: str | None, force: bool, question_workers: int) -> str:
    if question_workers < 1:
        raise ValueError("--question-workers must be a positive integer")
    item_id = item["id"]
    case_root = case_root_for_item(dataset, settings, item)
    standard_path = case_root / "trajectories" / "standard_trajectory.json"
    evidence_path = case_root / "evidence" / "evidence.json"

    answer_cfg = settings.llm_for("answer")
    model_name = model or answer_cfg.model
    model_root = case_root / "trajectories" / "model_perception_trajectory" / model_name
    trajectory_path = model_root / "model_perception_trajectory.json"
    report_path = model_root / "perception_report.json"
    cache_dir = case_root / "cache" / "stage1_perception" / model_name / "verifier"

    if not standard_path.is_file():
        raise FileNotFoundError(f"Standard trajectory does not exist: {standard_path}")
    if not evidence_path.is_file():
        raise FileNotFoundError(f"Evidence does not exist: {evidence_path}")
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Model perception trajectory does not exist: {trajectory_path}")
    if not force and report_path.is_file():
        log(f"[{dataset}-perception-eval][{item_id}][resume] model={model_name} report found; skipped")
        return "skipped"

    standard = read_json(standard_path)
    evidence = read_json(evidence_path)
    trajectory = read_json(trajectory_path)
    model_answers = model_answers_by_source(trajectory)
    verifier_cfg = settings.llm_for("verifier")

    with ChatClient(
        api_key=verifier_cfg.api_key,
        base_url=verifier_cfg.base_url,
        model=verifier_cfg.model,
        log_prefix=f"[{dataset}-perception-eval][{item_id}]",
    ) as verifier:
        evaluator = PerceptionEvaluator(
            verifier=verifier,
            standard=standard,
            evidence=evidence,
            cache_dir=cache_dir,
            report_path=report_path,
        )
        items = []
        for stage in sorted(standard.get("stages", []), key=lambda item: item.get("order", 0)):
            for qa in stage.get("qa_pairs", []):
                if not should_evaluate(dataset, stage, qa):
                    continue
                key = (stage["stage_id"], int(qa.get("source_turn_id", 0)))
                answer = model_answers.get(key, "")
                if not answer:
                    raise ValueError(f"Missing model answer for {stage['stage_id']} source_turn_id={qa.get('source_turn_id')}")
                items.append((stage, qa, answer))

        def run_item(item: tuple[dict, dict, str]) -> dict:
            stage, qa, answer = item
            record = evaluator.evaluate(stage, qa, answer)
            log(
                f"[{dataset}-perception-eval][{item_id}][{model_name}]"
                f"[question-evaluated] stage={stage['stage_id']} "
                f"source_turn_id={qa.get('source_turn_id')} "
                f"f1={record['metrics'].get('f1')}"
            )
            return record

        if question_workers == 1 or len(items) <= 1:
            records = [run_item(item) for item in items]
        else:
            records = []
            with ThreadPoolExecutor(max_workers=question_workers) as executor:
                futures = [executor.submit(run_item, item) for item in items]
                for future in as_completed(futures):
                    records.append(future.result())

        stage_order = {stage["stage_id"]: stage.get("order", 0) for stage in standard.get("stages", [])}
        evaluator.records = sorted(records, key=lambda item: (stage_order.get(item["stage_id"], 0), item["source_turn_id"]))
        evaluator.write_report()

    log(f"[{dataset}-perception-eval][{item_id}][done] model={model_name} output={report_path}")
    return "completed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model-perception trajectories")
    parser.add_argument("--dataset", choices=("patient", "report"), default="patient", help="Benchmark subset to process")
    add_batch_arguments(parser)
    parser.add_argument("--model", default=None, help="Override ANSWER_OPENAI_MODEL for locating saved trajectories")
    parser.add_argument("--question-workers", type=int, default=1, help="Questions scored concurrently within each patient/report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    items = selected_reports(REPORT_PDF_DIR, args.all, args.limit) if args.dataset == "report" else selected_patients(settings.dataset_json, args.all, args.limit)

    def worker(item: dict) -> str:
        return evaluate_case(args.dataset, item, settings, args.model, args.force, args.question_workers)

    return run_patient_batch(items, args.num_workers, f"{args.dataset}-perception-eval", worker)


if __name__ == "__main__":
    raise SystemExit(main())
