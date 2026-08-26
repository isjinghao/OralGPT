"""Evaluate generated report benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import get_settings
from utils.batch_utils import add_batch_arguments, log, run_patient_batch, selected_reports
from report_pipeline.paths import REPORT_OUTPUT_ROOT as OUTPUT_ROOT, REPORT_PDF_DIR as PDF_DIR
from step4_evaluation.memory import available_methods
from step4_evaluation.run_step4 import parse_csv, run_patient, trajectory_completed

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated report benchmarks")
    add_batch_arguments(parser)
    parser.add_argument("--trajectories", type=parse_csv, default=["standard_trajectory"])
    parser.add_argument("--methods", type=parse_csv, default=["full_context_memory"])
    parser.add_argument("--answer-model", default=None, help="Override ANSWER_OPENAI_MODEL for this run")
    parser.add_argument("--answer-output-model", default=None, help="Write results under this answer-model folder")
    parser.add_argument("--answer-base-url", default=None, help="Override ANSWER_OPENAI_BASE_URL for this run")
    parser.add_argument("--answer-workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--score-workers", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--method-workers", type=int, default=1)
    parser.add_argument("--phase", choices=("answers", "scoring"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    unknown_methods = sorted(set(args.methods) - set(available_methods()))
    if unknown_methods:
        raise ValueError(f"Unknown memory methods: {unknown_methods}")

    settings = get_settings()
    answer_request_model = args.answer_model or settings.llm_for("answer").model
    answer_model = args.answer_output_model or answer_request_model
    reports = selected_reports(PDF_DIR, args.all, args.limit)

    def worker(report: dict) -> str:
        patient_id = report["id"]
        out = OUTPUT_ROOT / report["name"]
        if not args.force and all(
            trajectory_completed(
                out,
                trajectory,
                args.methods,
                answer_model,
                args.phase,
            )
            for trajectory in args.trajectories
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
            answer_request_model=answer_request_model,
            answer_base_url=args.answer_base_url,
            force=args.force,
            answer_workers=args.answer_workers,
            score_workers=args.score_workers,
            method_workers=args.method_workers,
            phase=args.phase,
        )
        return "completed"

    return run_patient_batch(reports, args.num_workers, "evaluation", worker)


if __name__ == "__main__":
    raise SystemExit(main())
