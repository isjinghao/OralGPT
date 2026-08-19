"""Generate report evidence graphs, benchmark tasks, and rubrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import get_settings
from utils.batch_utils import add_batch_arguments, log, run_patient_batch, selected_reports
from utils.json_utils import read_json
from llm_client import build_client
from report_pipeline.paths import REPORT_OUTPUT_ROOT as OUTPUT_ROOT, REPORT_PDF_DIR as PDF_DIR, REPORT_ROOT as ROOT
from step2_evidence.pipeline import run_evidence_and_graph
from step3_tasks.run_step3 import completed as step3_completed
from step3_tasks.run_step3 import run_patient as run_step3

PROMPT_DIR = ROOT / "report_pipeline" / "prompts"
EVIDENCE_PROMPT = PROMPT_DIR / "evidence_extraction.yaml"


def step01_completed(out: Path) -> bool:
    return (out / "timeline.extracted.json").is_file() and (
        out / "trajectories" / "standard_trajectory.json"
    ).is_file()


def step2_completed(out: Path) -> bool:
    return all(
        path.is_file()
        for path in (
            out / "evidence" / "evidence.json",
            out / "graph" / "evidence_graph.json",
            out / "graph" / "evidence_graph.html",
        )
    )


def run_step2(out: Path, patient_id: str, settings, stage_workers: int = 2) -> None:
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
    parser = argparse.ArgumentParser(description="Run report Step2 evidence and Step3 benchmark generation")
    add_batch_arguments(parser)
    parser.add_argument("--stage-workers", type=int, default=2)
    parser.add_argument("--task-workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    reports = selected_reports(PDF_DIR, args.all, args.limit)

    def worker(report: dict) -> str:
        patient_id = report["id"]
        out = OUTPUT_ROOT / report["name"]
        if not step01_completed(out):
            log(f"[benchmark][{patient_id}][step2-step3] skipped: Step0–Step1 incomplete")
            return "skipped"
        if not args.force and step2_completed(out) and step3_completed(out):
            log(f"[benchmark][{patient_id}][step2-step3/resume] completed outputs found; skipped")
            return "skipped"
        if args.force or not step2_completed(out):
            run_step2(out, patient_id, settings, args.stage_workers)
        else:
            log(f"[benchmark][{patient_id}][step2/resume] completed outputs found; skipped")
        run_step3(out, patient_id, settings, PROMPT_DIR, args.task_workers)
        return "completed"

    return run_patient_batch(reports, args.num_workers, "benchmark", worker)


if __name__ == "__main__":
    raise SystemExit(main())
