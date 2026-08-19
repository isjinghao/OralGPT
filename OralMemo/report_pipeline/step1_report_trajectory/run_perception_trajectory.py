"""Generate model-perception trajectories for report benchmarks."""
from __future__ import annotations

import argparse
from pathlib import Path

from config import get_settings
from llm_client import ChatClient
from report_pipeline.paths import REPORT_OUTPUT_ROOT as OUTPUT_ROOT, REPORT_PDF_DIR as PDF_DIR
from step1_patient_trajectory.perception_evaluation import PerceptionEvaluator
from step2_evidence.run_perception_trajectory import format_profile, generate_answer
from utils.batch_utils import add_batch_arguments, log, run_patient_batch, selected_reports
from utils.json_utils import read_json, write_json

def observation_record(stage: dict, qa: dict, answer: str) -> dict:
    return {
        "stage_id": stage["stage_id"],
        "source_turn_id": qa["source_turn_id"],
        "question": qa.get("human", ""),
        "answer": answer,
    }


def generate_trajectory(
    standard: dict,
    client: ChatClient,
    image_root: Path,
    cache_dir: Path,
    force: bool,
    evaluator: PerceptionEvaluator,
) -> dict:
    text_memory: list[dict] = []
    image_memory: list[dict] = []
    generated_stages: list[dict] = []

    for stage in sorted(standard["stages"], key=lambda item: item["order"]):
        generated = {key: value for key, value in stage.items() if key != "qa_pairs"}
        qa_pairs = []
        for qa in stage["qa_pairs"]:
            images = qa.get("image_paths", [])
            if qa["role"] != "observation" or not images:
                qa_pairs.append(dict(qa))
                if qa["role"] == "observation":
                    text_memory.append(observation_record(stage, qa, qa["assistant"]))
                continue

            answer = generate_answer(
                client=client,
                profile=format_profile([]),
                text_memory=text_memory,
                memory=image_memory,
                stage=stage,
                qa=qa,
                image_root=image_root,
                cache_dir=cache_dir,
                force=force,
            )
            qa_pairs.append({**qa, "assistant": answer})
            evaluator.add_and_write(stage, qa, answer)
            image_memory.append(observation_record(stage, qa, answer))

        generated["qa_pairs"] = qa_pairs
        generated_stages.append(generated)

    stage_order = {stage["stage_id"]: stage["order"] for stage in standard["stages"]}
    evaluator.records.sort(key=lambda item: (stage_order[item["stage_id"]], item["source_turn_id"]))
    evaluator.write_report()
    result = {
        **{key: value for key, value in standard.items() if key not in {"trajectory_id", "trajectory_type", "stages"}},
        "trajectory_id": f"{standard['patient_id']}__model_perception_trajectory",
        "trajectory_type": "model_perception_trajectory",
        "stages": generated_stages,
    }
    return result


def run_report(report: dict, settings, model: str | None, base_url: str | None, force: bool) -> str:
    patient_id = report["id"]
    case_root = OUTPUT_ROOT / report["name"]
    standard_path = case_root / "trajectories" / "standard_trajectory.json"
    evidence_path = case_root / "evidence" / "evidence.json"

    answer_cfg = settings.llm_for("answer")
    model_name = model or answer_cfg.model
    model_base_url = base_url or answer_cfg.base_url
    model_root = case_root / "trajectories" / "model_perception_trajectory" / model_name
    output_path = model_root / "model_perception_trajectory.json"
    report_path = model_root / "perception_report.json"
    cache_dir = case_root / "cache" / "stage1_perception" / model_name

    if not force and output_path.is_file() and report_path.is_file():
        log(f"[report-perception][{patient_id}][resume] model={model_name} outputs found; skipped")
        return "skipped"

    standard = read_json(standard_path)
    evidence = read_json(evidence_path)
    verifier_cfg = settings.llm_for("verifier")
    with (
        ChatClient(
            api_key=answer_cfg.api_key,
            base_url=model_base_url,
            model=model_name,
            log_prefix=f"[report-perception][{patient_id}][{model_name}]",
        ) as client,
        ChatClient(
            api_key=verifier_cfg.api_key,
            base_url=verifier_cfg.base_url,
            model=verifier_cfg.model,
            log_prefix=f"[report-perception-verifier][{patient_id}]",
        ) as verifier,
    ):
        evaluator = PerceptionEvaluator(
            verifier=verifier,
            standard=standard,
            evidence=evidence,
            cache_dir=cache_dir / "verifier",
            report_path=report_path,
        )
        result = generate_trajectory(
            standard=standard,
            client=client,
            image_root=settings.bench_root,
            cache_dir=cache_dir,
            force=force,
            evaluator=evaluator,
        )
        write_json(output_path, result)
    log(f"[report-perception][{patient_id}][done] model={model_name} output={output_path}")
    return "completed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report model-perception trajectories")
    add_batch_arguments(parser)
    parser.add_argument("--model", default=None, help="Override ANSWER_OPENAI_MODEL for this run")
    parser.add_argument("--base-url", default=None, help="Override ANSWER_OPENAI_BASE_URL for this run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    reports = selected_reports(PDF_DIR, args.all, args.limit)

    def worker(report: dict) -> str:
        return run_report(report, settings, args.model, args.base_url, args.force)

    return run_patient_batch(reports, args.num_workers, "report-perception", worker)


if __name__ == "__main__":
    raise SystemExit(main())
