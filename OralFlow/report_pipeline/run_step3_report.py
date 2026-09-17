"""Generate benchmark tasks and rubrics from report evidence graphs."""

from __future__ import annotations

import argparse

from config import get_settings
from report_pipeline.paths import (
    REPORT_OUTPUT_ROOT as OUTPUT_ROOT,
    REPORT_PDF_DIR as PDF_DIR,
    REPORT_ROOT as ROOT,
    step01_completed,
    step2_completed,
)
from step3_tasks.run_step3 import completed as step3_completed
from step3_tasks.run_step3 import run_patient as run_step3
from utils.batch_utils import add_batch_arguments, log, run_patient_batch, selected_reports

PROMPT_DIR = ROOT / "report_pipeline" / "prompts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report Step3 benchmark tasks and rubrics")
    add_batch_arguments(parser)
    parser.add_argument("--task-workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    reports = selected_reports(PDF_DIR, args.all, args.limit)

    def worker(report: dict) -> str:
        patient_id = report["id"]
        out = OUTPUT_ROOT / report["name"]
        if not step01_completed(out) or not step2_completed(out):
            log(f"[benchmark][{patient_id}][step3] skipped: Step0-Step2 incomplete")
            return "skipped"
        if not args.force and step3_completed(out):
            log(f"[benchmark][{patient_id}][step3/resume] completed outputs found; skipped")
            return "skipped"
        run_step3(out, patient_id, settings, PROMPT_DIR, args.task_workers)
        return "completed"

    return run_patient_batch(reports, args.num_workers, "benchmark", worker)


if __name__ == "__main__":
    raise SystemExit(main())
