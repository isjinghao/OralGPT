from __future__ import annotations

from pathlib import Path

from config import Settings
from utils.batch_utils import log, patient_output_root
from utils.json_utils import write_json
from llm_client import ChatClient
from step1_patient_trajectory.dataset import build_source_turns
from step1_patient_trajectory.stages import build_patient_stages
from step1_patient_trajectory.trajectories import (
    NOISE_VARIANTS,
    build_missing_modality_variants,
    build_noisy_variant,
    build_standard_trajectory,
)
from step2_evidence.evidence import extract_all_evidence
from step2_evidence.graph import build_evidence_graph
from step2_evidence.visualize_graph import render_html, render_png


def run_evidence_and_graph(
    standard: dict,
    out: Path,
    settings: Settings,
    client: ChatClient,
    *,
    log_prefix: str,
    stage_workers: int = 2,
    prompt_path: Path | None = None,
) -> None:
    log(f"{log_prefix}[step2/evidence] started")
    extract_kwargs = {"cache_dir": out / "cache", "log_prefix": log_prefix, "stage_workers": stage_workers}
    if prompt_path is not None:
        extract_kwargs["prompt_path"] = prompt_path
    evidence = extract_all_evidence(client, standard, **extract_kwargs)
    evidence_path = out / "evidence" / "evidence.json"
    write_json(evidence_path, evidence)
    log(f"{log_prefix}[step2/evidence] completed count={len(evidence['evidence'])}")

    log(f"{log_prefix}[step2/graph] started")
    graph = build_evidence_graph(
        evidence_path,
        client=client,
        cache_dir=out / "cache",
        max_edges=settings.graph_max_edges,
        log_prefix=log_prefix,
    )
    graph_dir = out / "graph"
    write_json(graph_dir / "evidence_graph.json", graph)
    html_path = graph_dir / "evidence_graph.html"
    render_html(graph, evidence["evidence"], standard["stages"], html_path)
    try:
        render_png(html_path, graph_dir / "evidence_graph.png")
        artifacts = "json,html,png"
    except Exception as exc:
        log(f"{log_prefix}[step2/graph] png skipped: {type(exc).__name__}: {exc}")
        artifacts = "json,html"
    log(f"{log_prefix}[step2/graph] completed edges={len(graph['edges'])} artifacts={artifacts}")


def process_patient(item: dict, settings: Settings, client: ChatClient, stage_workers: int = 2) -> None:
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
    for trajectory_type, noise_count in NOISE_VARIANTS:
        noisy = build_noisy_variant(standard, trajectory_type, noise_count)
        write_json(out / "trajectories" / trajectory_type / f"{trajectory_type}.json", noisy)
    log(
        f"[benchmark][{patient_id}][step1/done] stages={len(standard['stages'])} "
        f"variants={len(variants) + len(NOISE_VARIANTS)}"
    )

    run_evidence_and_graph(
        standard,
        out,
        settings,
        client,
        log_prefix=f"[benchmark][{patient_id}]",
        stage_workers=stage_workers,
    )

