"""Run report PDF ingestion and longitudinal trajectory construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import get_settings
from utils.batch_utils import add_batch_arguments, log, run_patient_batch, selected_reports
from utils.json_utils import write_json
from llm_client import ChatClient, build_client
from report_pipeline.paths import REPORT_OUTPUT_ROOT as OUTPUT_ROOT, REPORT_PDF_DIR as PDF_DIR, REPORT_ROOT as ROOT
from report_pipeline.step0_ingest.pdf_extract import extract_pdf
from report_pipeline.step0_ingest.timeline_llm import extract_timeline, repair_timeline
from report_pipeline.step0_ingest.verify_llm import verify_timeline
from report_pipeline.step1_report_trajectory.qa_render import normalize_timepoints, render_turns, sanitize_timeline
from report_pipeline.step1_report_trajectory.report_dataset import build_report_dataset_entry
from report_pipeline.step1_report_trajectory.report_stages import build_report_stages
from step1_patient_trajectory.trajectories import build_standard_trajectory

def extract_with_feedback(
    extract_client: ChatClient,
    verifier_client: ChatClient,
    raw_dir: Path,
    figures: list[dict],
    images_map: dict,
    max_iters: int,
    patient_id: str,
    history_path: Path | None = None,
) -> tuple[dict, list[dict], bool]:
    history: list[dict] = []
    feedback: list[dict] = []
    previous_issues: list[dict] = []
    timeline: dict = {}
    seen_timelines: list[dict] = []
    prefix = f"[benchmark][{patient_id}]"

    for iteration in range(1, max_iters + 1):
        if iteration == 1:
            log(f"{prefix}[step0/extract] iteration={iteration}/{max_iters}")
            timeline = extract_timeline(extract_client, raw_dir, figures)
        else:
            log(f"{prefix}[step0/repair] iteration={iteration}/{max_iters}")
            timeline = repair_timeline(extract_client, raw_dir, figures, timeline, feedback)

        timeline = sanitize_timeline(timeline, images_map)
        log(f"{prefix}[step0/verify] iteration={iteration}/{max_iters}")
        try:
            render_turns(normalize_timepoints(timeline), images_map)
        except (KeyError, TypeError, ValueError) as exc:
            verification = {
                "passed": False,
                "issues": [{
                    "severity": "high",
                    "location": "timeline structure",
                    "problem": str(exc),
                    "source_evidence": "",
                    "suggested_fix": (
                        "Repair the implicated timepoint and any later stage boundary together; "
                        "once followup starts, later interventions remain followup."
                    ),
                }],
            }
        else:
            verification = verify_timeline(verifier_client, raw_dir, timeline, figures, previous_issues)

        feedback = [
            issue for issue in verification["issues"]
            if issue["severity"] in {"high", "medium"}
        ]
        previous_issues.extend(feedback)
        verification["passed"] = not feedback
        n_high = sum(issue["severity"] == "high" for issue in feedback)
        history.append(
            {
                "iteration": iteration,
                "passed": verification["passed"],
                "n_issues": len(verification["issues"]),
                "n_high": n_high,
                "n_actionable": len(feedback),
                "verification": verification,
            }
        )
        if history_path:
            write_json(history_path, history)
        log(
            f"{prefix}[step0/verify] passed={verification['passed']} "
            f"issues={len(verification['issues'])} high={n_high} actionable={len(feedback)}"
        )
        if verification["passed"]:
            return timeline, history, True
        if timeline in seen_timelines:
            log(f"{prefix}[step0/verify] repeated timeline detected; stopping repair")
            break
        seen_timelines.append(timeline)

    return timeline, history, False


def run_report(report: dict, settings, args: argparse.Namespace) -> None:
    name = report["name"]
    patient_id = report["id"]
    pdf_path = report["pdf_path"]
    out_dir = OUTPUT_ROOT / name
    raw_dir = out_dir / "raw"
    images_dir = out_dir / "images"
    timeline_path = out_dir / "timeline.extracted.json"
    verification_path = out_dir / "verification_report.json"
    trajectory_path = out_dir / "trajectories" / "standard_trajectory.json"
    dataset_path = out_dir / "dataset_entry.json"
    prefix = f"[benchmark][{patient_id}]"

    ingest_files = [raw_dir / name for name in ("fulltext.json", "tables.json", "captions.json")]
    if args.force or not all(path.is_file() for path in ingest_files):
        log(f"{prefix}[step0/ingest] pdf={pdf_path.name}")
        summary = extract_pdf(pdf_path, raw_dir, images_dir, rel_base=ROOT)
        images_map = summary["images_map"]
        write_json(raw_dir / "captions.json", images_map)
        write_json(raw_dir / "unmapped_images.json", summary["unmapped_images"])
        log(
            f"{prefix}[step0/ingest] pages={summary['n_pages']} images={summary['n_images_kept']} "
            f"tables={summary['n_tables']} captions={len(images_map)} "
            f"unmapped={len(summary['unmapped_images'])}"
        )
    else:
        log(f"{prefix}[step0/ingest] completed outputs found; skipped")
        images_map = json.loads((raw_dir / "captions.json").read_text(encoding="utf-8"))

    reuse_timeline = not args.force and timeline_path.is_file() and verification_path.is_file()
    if reuse_timeline:
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        verification_history = json.loads(verification_path.read_text(encoding="utf-8"))
        try:
            if not verification_history or not verification_history[-1]["passed"]:
                raise ValueError("cached verification did not pass")
            timeline = sanitize_timeline(timeline, images_map)
            render_turns(normalize_timepoints(timeline), images_map)
        except (KeyError, TypeError, ValueError) as exc:
            log(f"{prefix}[step0/timeline] cached timeline invalid; regenerating: {exc}")
            reuse_timeline = False

    if reuse_timeline:
        log(f"{prefix}[step0/timeline] completed valid outputs found; skipped")
    else:
        timeline_path.unlink(missing_ok=True)
        captions = [
            {"figure": figure, "caption": entry.get("caption", "")}
            for figure, entry in images_map.items()
        ]
        with (
            build_client(settings, "benchmark", patient_id, log_prefix="[benchmark]", model=args.model) as extract_client,
            build_client(settings, "verifier", patient_id, log_prefix="[benchmark]") as verifier_client,
        ):
            timeline, verification_history, passed = extract_with_feedback(
                extract_client,
                verifier_client,
                raw_dir,
                captions,
                images_map,
                args.max_iters,
                patient_id,
                verification_path,
            )
        write_json(verification_path, verification_history)
        if not passed:
            raise ValueError(
                f"Timeline verification failed after {args.max_iters} iterations "
                f"with {verification_history[-1]['n_actionable']} actionable issues"
            )
        write_json(timeline_path, timeline)

    if reuse_timeline and trajectory_path.is_file() and dataset_path.is_file():
        log(f"{prefix}[step1/trajectory] completed outputs found; skipped")
        return

    log(f"{prefix}[step1/trajectory] building standard trajectory")
    patient = {"patient_id": patient_id, "name": name, "group": "report"}
    normalized = normalize_timepoints(timeline)
    rendered = render_turns(normalized, images_map)
    stages = build_report_stages(normalized, rendered, patient)
    standard = build_standard_trajectory(stages)
    dataset_entry = build_report_dataset_entry(
        standard,
        patient,
        pdf_path.relative_to(ROOT).as_posix(),
    )
    write_json(trajectory_path, standard)
    write_json(dataset_path, dataset_entry)
    evaluation_count = sum(
        qa["role"] == "evaluation"
        for stage in standard["stages"]
        for qa in stage["qa_pairs"]
    )
    log(
        f"{prefix}[step1/done] stages={len(standard['stages'])} qa={dataset_entry['num_qa_pairs']} "
        f"images={dataset_entry['num_images']} evaluation={evaluation_count}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run report Step0 ingestion and Step1 trajectory construction")
    add_batch_arguments(parser)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    reports = selected_reports(PDF_DIR, args.all, args.limit)

    def worker(report: dict) -> str:
        run_report(report, settings, args)
        return "completed"

    return run_patient_batch(reports, args.num_workers, "benchmark", worker)


if __name__ == "__main__":
    raise SystemExit(main())
