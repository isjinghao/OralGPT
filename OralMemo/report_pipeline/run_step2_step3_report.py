"""Generate report evidence graphs, benchmark tasks, and rubrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from batch_utils import add_batch_arguments, log, run_patient_batch, selected_reports
from config import get_settings
from step2_evidence.evidence import extract_all_evidence
from step2_evidence.graph import build_evidence_graph
from step2_evidence.pipeline import build_client, write_json
from step2_evidence.visualize_graph import render_html, render_png
from step3_tasks.run_step3 import completed as step3_completed
from step3_tasks.run_step3 import run_patient as run_step3

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "reports" / "pdf"
OUTPUT_ROOT = ROOT / "outputs" / "report"
PROMPT_DIR = ROOT / "report_pipeline" / "prompts"
EVIDENCE_PROMPT = PROMPT_DIR / "evidence_extraction.yaml"


def step2_completed(out: Path) -> bool:
    return all(
        path.is_file()
        for path in (
            out / "evidence" / "evidence.json",
            out / "graph" / "evidence_graph.json",
            out / "graph" / "evidence_graph.html",
        )
    )


def run_step2(out: Path, patient_id: str, settings) -> None:
    prefix = f"[benchmark][{patient_id}]"
    standard = json.loads(
        (out / "trajectories" / "standard_trajectory.json").read_text(encoding="utf-8")
    )
    client = build_client(settings, patient_id)

    log(f"{prefix}[step2/evidence] started")
    evidence = extract_all_evidence(
        client,
        standard,
        cache_dir=out / "cache",
        log_prefix=prefix,
        prompt_path=EVIDENCE_PROMPT,
    )
    evidence_path = out / "evidence" / "evidence.json"
    write_json(evidence_path, evidence)
    log(f"{prefix}[step2/evidence] completed count={len(evidence['evidence'])}")

    log(f"{prefix}[step2/graph] started")
    graph = build_evidence_graph(
        evidence_path,
        client=client,
        cache_dir=out / "cache",
        max_edges=settings.graph_max_edges,
        log_prefix=prefix,
    )
    graph_dir = out / "graph"
    write_json(graph_dir / "evidence_graph.json", graph)
    html_path = graph_dir / "evidence_graph.html"
    render_html(graph, evidence["evidence"], standard["stages"], html_path)
    try:
        render_png(html_path, graph_dir / "evidence_graph.png")
        artifacts = "json,html,png"
    except Exception as exc:
        log(f"{prefix}[step2/graph] png skipped: {type(exc).__name__}: {exc}")
        artifacts = "json,html"
    log(f"{prefix}[step2/graph] completed edges={len(graph['edges'])} artifacts={artifacts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run report Step2 evidence and Step3 benchmark generation")
    add_batch_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    reports = selected_reports(PDF_DIR, args.all, args.limit)

    def worker(report: dict) -> str:
        patient_id = report["id"]
        out = OUTPUT_ROOT / report["name"]
        if not args.force and step2_completed(out) and step3_completed(out):
            log(f"[benchmark][{patient_id}][step2-step3/resume] completed outputs found; skipped")
            return "skipped"
        if args.force or not step2_completed(out):
            run_step2(out, patient_id, settings)
        else:
            log(f"[benchmark][{patient_id}][step2/resume] completed outputs found; skipped")
        run_step3(out, patient_id, settings, PROMPT_DIR)
        return "completed"

    return run_patient_batch(reports, args.num_workers, "benchmark", worker)


if __name__ == "__main__":
    raise SystemExit(main())
