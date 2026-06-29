from __future__ import annotations

import json
from pathlib import Path

from bench.config import Settings
from bench.llm_client import ChatClient
from bench.step1_patient_trajectory.dataset import build_source_turns
from bench.step1_patient_trajectory.stages import build_patient_stages
from bench.step1_patient_trajectory.trajectories import (
    build_long_noisy_variant,
    build_missing_modality_variants,
    build_standard_trajectory,
)
from bench.step2_evidence.evidence import extract_all_evidence


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def patient_output_root(settings: Settings, patient_id: str) -> Path:
    group, sep, name = patient_id.partition("__")
    if sep:
        return settings.bench_root / "outputs" / group / name
    return settings.bench_root / "outputs" / patient_id


def process_patient(item: dict, settings: Settings, client: ChatClient):
    patient_id = item["id"]
    out = patient_output_root(settings, patient_id)

    # Step1: 轨迹 / 阶段
    source_turns = build_source_turns(item)
    patient_stages = build_patient_stages(source_turns)
    write_json(out / "stages" / "patient_stages.json", patient_stages)

    standard = build_standard_trajectory(patient_stages)
    write_json(out / "trajectories" / "standard_trajectory.json", standard)

    for variant in build_missing_modality_variants(standard):
        write_json(out / "variants" / f"{variant['trajectory_type']}.json", variant)
    write_json(out / "variants" / "long_noisy.json", build_long_noisy_variant(standard))

    # Step2: 原子证据抽取(LLM)
    evidence = extract_all_evidence(client, patient_stages, cache_dir=out / "cache")
    write_json(out / "evidence" / "evidence.json", evidence)


def build_client(settings: Settings) -> ChatClient:
    return ChatClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
    )
