from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from string import Template

import yaml

from batch_utils import log
from llm_client import ChatClient


PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "evidence_extraction.yaml"


def load_prompt_template(prompt_path: Path) -> Template:
    config = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    return Template(config["template"])


def canonical_modalities(stage: dict, item: dict) -> list[str]:
    # 只保留当前源阶段允许的模态；错误输出回退到阶段模态
    allowed = list(stage.get("modality", []))
    reported = [modality for modality in item.get("modality", []) or [] if modality in allowed]
    return reported or allowed


def slim_evidence(record: dict) -> dict:
    return {
        "evidence_id": record["evidence_id"],
        "source_turn_id": record["source_turn_id"],
        "introduced_stage": record["introduced_stage"],
        "modality": record.get("modality", []),
        "fact_text": record.get("fact_text", ""),
        "fact_type": record.get("fact_type", "other"),
        "clinical_dimension": record.get("clinical_dimension", "other"),
        "normalized": record.get("normalized", {}),
    }


def extract_stage_evidence(
    client: ChatClient,
    patient_id: str,
    stage: dict,
    prompt_template: Template,
) -> list[dict]:
    evidence = []
    used_counts = defaultdict(int)

    for turn in stage["qa_pairs"]:
        prompt = prompt_template.substitute(
            patient_id=patient_id,
            stage_id=stage["stage_id"],
            modalities=", ".join(stage["modality"]),
            qa_text=(
                f"[source_turn_id={turn['source_turn_id']}]\n"
                f"Question: {turn['human'].replace('<image>', '').strip()}\n"
                f"Answer: {turn['assistant'].strip()}"
            ),
        )
        result = client.complete_json(prompt, max_tokens=8000)
        raw_evidence = result.get("atomic_evidence", [])

        for item in raw_evidence:
            source_turn_id = item.get("source_turn_id") or turn["source_turn_id"]
            normalized = item.get("normalized") or {}
            field = normalized.get("field") or item.get("fact_type", "fact")
            key = f"{stage['stage_id']}_{source_turn_id}_{field}".lower()
            used_counts[key] += 1
            evidence_id = f"{patient_id}_{key}_{used_counts[key]:02d}".replace("__", "_")
            evidence.append(
                slim_evidence(
                    {
                        "evidence_id": evidence_id,
                        "source_turn_id": source_turn_id,
                        "introduced_stage": stage["stage_id"],
                        "modality": canonical_modalities(stage, item),
                        "fact_text": item.get("fact_text", ""),
                        "fact_type": item.get("fact_type", "other"),
                        "clinical_dimension": item.get("clinical_dimension", "other"),
                        "normalized": normalized,
                    }
                )
            )
    return evidence


def extract_all_evidence(
    client: ChatClient,
    patient_stages: dict,
    cache_dir: Path | None = None,
    log_prefix: str | None = None,
    prompt_path: Path = PROMPT_PATH,
) -> dict:
    prefix = log_prefix or f"[benchmark][{patient_stages['patient_id']}]"
    prompt_text = prompt_path.read_text(encoding="utf-8")
    prompt_template = load_prompt_template(prompt_path)
    all_evidence = []
    stages = patient_stages["stages"]
    for stage_index, stage in enumerate(stages, start=1):
        evidence_stage = {
            **stage,
            "qa_pairs": [
                turn for turn in stage["qa_pairs"]
                if turn["role"] == "observation"
            ],
        }
        cache_input = {
            "model": client.model,
            "prompt": prompt_text,
            "stage": evidence_stage,
        }
        cache_path = cache_dir / "evidence" / f"{stage['stage_id']}.json" if cache_dir else None
        cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path and cache_path.exists() else None
        if cached and cached.get("input") == cache_input:
            stage_evidence = cached["evidence"]
            log(
                f"{prefix}[step2/evidence] stage={stage_index}/{len(stages)} "
                f"id={stage['stage_id']} cache_hit count={len(stage_evidence)}"
            )
        else:
            log(
                f"{prefix}[step2/evidence] stage={stage_index}/{len(stages)} "
                f"id={stage['stage_id']} extracting"
            )
            stage_evidence = extract_stage_evidence(
                client,
                patient_stages["patient_id"],
                evidence_stage,
                prompt_template,
            )
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(
                        {"input": cache_input, "evidence": stage_evidence},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            log(
                f"{prefix}[step2/evidence] stage={stage_index}/{len(stages)} "
                f"id={stage['stage_id']} extracted count={len(stage_evidence)}"
            )
        all_evidence.extend(slim_evidence(e) for e in stage_evidence)
    return {
        "patient_id": patient_stages["patient_id"],
        "evidence": all_evidence,
    }
