from __future__ import annotations

import argparse
import json
from pathlib import Path

from step2_evidence.visualize_graph import render_html

BENCH_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    out_dir = BENCH_ROOT / "outputs" / "report" / args.name
    graph = json.loads((out_dir / "graph" / "evidence_graph.json").read_text(encoding="utf-8"))
    evidence = json.loads((out_dir / "evidence" / "evidence.json").read_text(encoding="utf-8"))["evidence"]
    stages = json.loads(
        (out_dir / "trajectories" / "standard_trajectory.json").read_text(encoding="utf-8")
    )["stages"]
    html_path = out_dir / "graph" / "evidence_graph.html"
    render_html(graph, evidence, stages, html_path)
    print(f"[viz] {args.name}: nodes={len(evidence)} edges={len(graph.get('edges', []))} -> {html_path}")


if __name__ == "__main__":
    main()
