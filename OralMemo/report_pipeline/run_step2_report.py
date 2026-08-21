"""Generate evidence graphs from report standard trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import get_settings
from llm_client import build_client
from report_pipeline.paths import (
    REPORT_OUTPUT_ROOT as OUTPUT_ROOT,
    REPORT_PDF_DIR as PDF_DIR,
    REPORT_ROOT as ROOT,
    step01_completed,
    step2_completed,
)
from step2_evidence.pipeline import run_evidence_and_graph
from utils.batch_utils import add_batch_arguments, log, run_patient_batch, selected_reports
from utils.json_utils import read_json

PROMPT_DIR = ROOT / "report_pipeline" / "prompts"
EVIDENCE_PROMPT = PROMPT_DIR / "evidence_extraction.yaml"


def run_step2(out: Path, patient_id: str, settings, stage_workers: int) -> None:
    standard = read_json(out / "trajectories" / "standard_trajectory.json")
    with build_client(settings, "benchmark", patient_id, log_prefix="[benchmark]") as client:
        run_evidence_and_graph(
            standard,
            out,
            settings,
            client,
            log_prefix=f"[benchmark][{patient_id}]",
            prompt_path=EVIDENCE_PROMPT,
            stage_workers=stage_workers,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report Step2 evidence graphs")
    add_batch_arguments(parser)
    parser.add_argument("--stage-workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    reports = selected_reports(PDF_DIR, args.all, args.limit)

    def worker(report: dict) -> str:
        patient_id = report["id"]
        out = OUTPUT_ROOT / report["name"]
        if not step01_completed(out):
            log(f"[benchmark][{patient_id}][step2] skipped: Step0-Step1 incomplete")
            return "skipped"
        if not args.force and step2_completed(out):
            log(f"[benchmark][{patient_id}][step2/resume] completed outputs found; skipped")
            return "skipped"
        run_step2(out, patient_id, settings, args.stage_workers)
        return "completed"

    return run_patient_batch(reports, args.num_workers, "benchmark", worker)


if __name__ == "__main__":
    raise SystemExit(main())
