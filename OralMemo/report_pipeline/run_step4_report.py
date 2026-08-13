"""Evaluate generated report benchmarks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from batch_utils import add_batch_arguments, log, run_patient_batch, selected_reports
from config import get_settings
from step4_evaluation.memory import available_methods
from step4_evaluation.run_step4 import parse_csv, run_patient, trajectory_completed

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "reports" / "pdf"
OUTPUT_ROOT = ROOT / "outputs" / "report"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated report benchmarks")
    add_batch_arguments(parser)
    parser.add_argument("--methods", type=parse_csv, default=["full_context_memory"])
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--answer-model", default=None, help="Override ANSWER_OPENAI_MODEL for this run")
    parser.add_argument("--answer-base-url", default=None, help="Override ANSWER_OPENAI_BASE_URL for this run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.answer_model:
        os.environ["ANSWER_OPENAI_MODEL"] = args.answer_model
    if args.answer_base_url:
        os.environ["ANSWER_OPENAI_BASE_URL"] = args.answer_base_url
    unknown_methods = sorted(set(args.methods) - set(available_methods()))
    if unknown_methods:
        raise ValueError(f"Unknown memory methods: {unknown_methods}")

    settings = get_settings()
    answer_model = args.answer_model or settings.llm_for("answer").model
    reports = selected_reports(PDF_DIR, args.all, args.limit)

    def worker(report: dict) -> str:
        patient_id = report["id"]
        out = OUTPUT_ROOT / report["name"]
        if not args.force and trajectory_completed(
            out,
            "standard_trajectory",
            args.methods,
            args.multimodal,
            answer_model,
        ):
            log(f"[evaluation][{patient_id}][step4/resume] completed outputs found; skipped")
            return "skipped"
        run_patient(
            out,
            patient_id,
            settings,
            ["standard_trajectory"],
            args.methods,
            args.multimodal,
            answer_model=answer_model,
            answer_base_url=args.answer_base_url,
        )
        return "completed"

    return run_patient_batch(reports, args.num_workers, "evaluation", worker)


if __name__ == "__main__":
    raise SystemExit(main())
