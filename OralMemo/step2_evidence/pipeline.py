from __future__ import annotations

import json
from pathlib import Path

from batch_utils import log, patient_output_root
from config import Settings
from llm_client import ChatClient
from step1_patient_trajectory.dataset import build_source_turns
from step1_patient_trajectory.stages import build_patient_stages
from step1_patient_trajectory.trajectories import (
    build_long_noisy_variant,
    build_missing_modality_variants,
    build_standard_trajectory,
)
from step2_evidence.evidence import extract_all_evidence
from step2_evidence.graph import build_evidence_graph
from step2_evidence.visualize_graph import render_html, render_png


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def process_patient(item: dict, settings: Settings, client: ChatClient) -> None:
    patient_id = item["id"]
    out = patient_output_root(settings.bench_root, patient_id)

    log(f"[benchmark][{patient_id}][step1/start] building trajectories")
    source_turns = build_source_turns(item)
    stages = build_patient_stages(source_turns)
    standard = build_standard_trajectory(stages)
    write_json(out / "trajectories" / "standard_trajectory.json", standard)
    variants = build_missing_modality_variants(standard)
    for variant in variants:
        trajectory_type = variant["trajectory_type"]
        write_json(out / "trajectories" / trajectory_type / f"{trajectory_type}.json", variant)
    write_json(out / "trajectories" / "long_noisy" / "long_noisy.json", build_long_noisy_variant(standard))
    log(f"[benchmark][{patient_id}][step1/done] stages={len(standard['stages'])} variants={len(variants) + 1}")

    log(f"[benchmark][{patient_id}][step2/evidence] started")
    evidence = extract_all_evidence(
        client,
        standard,
        cache_dir=out / "cache",
        log_prefix=f"[benchmark][{patient_id}]",
    )
    evidence_path = out / "evidence" / "evidence.json"
    write_json(evidence_path, evidence)
    log(f"[benchmark][{patient_id}][step2/evidence] completed count={len(evidence['evidence'])}")

    log(f"[benchmark][{patient_id}][step2/graph] started")
    graph = build_evidence_graph(
        evidence_path,
        client=client,
        cache_dir=out / "cache",
        max_edges=settings.graph_max_edges,
        log_prefix=f"[benchmark][{patient_id}]",
    )
    graph_dir = out / "graph"
    write_json(graph_dir / "evidence_graph.json", graph)
    html_path = graph_dir / "evidence_graph.html"
    render_html(graph, evidence["evidence"], standard["stages"], html_path)
    try:
        render_png(html_path, graph_dir / "evidence_graph.png")
        artifacts = "json,html,png"
    except Exception as exc:
        log(f"[benchmark][{patient_id}][step2/graph] png skipped: {type(exc).__name__}: {exc}")
        artifacts = "json,html"
    log(
        f"[benchmark][{patient_id}][step2/graph] completed edges={len(graph['edges'])} "
        f"artifacts={artifacts}"
    )


def build_client(settings: Settings, patient_id: str) -> ChatClient:
    cfg = settings.llm_for("benchmark")
    return ChatClient(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=cfg.model,
        log_prefix=f"[benchmark][{patient_id}]",
    )
