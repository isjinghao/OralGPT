from pathlib import Path


REPORT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PDF_DIR = REPORT_ROOT / "reports" / "pdf"
REPORT_OUTPUT_ROOT = REPORT_ROOT / "outputs" / "report"


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
