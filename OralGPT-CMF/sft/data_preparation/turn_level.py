#!/usr/bin/env python3
"""Create turn-level OralCMF SFT data with gold history and patient context."""

from __future__ import annotations

import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sft.data_preparation.patient_context import link_media, write_json
from common.conversation import rollout_system_context

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/turn_level"
PATIENT_STAGING = ROOT / ".sft_turn_level_patient_context"
SPLITS = ("train", "validation", "test", "human_evaluation")


def build_patient_context_staging() -> tuple[Any, Path]:
    from sft.data_preparation import patient_context

    raw_staging = patient_context.build_staging_dataset()
    if PATIENT_STAGING.exists():
        shutil.rmtree(PATIENT_STAGING)
    PATIENT_STAGING.mkdir(parents=True)
    return patient_context, raw_staging


def pairs(patient: dict[str, Any]) -> list[tuple[dict[str, str], dict[str, str], list[str]]]:
    messages = patient["conversations"]
    if not messages or messages[0].get("from") != "system" or (len(messages) - 1) % 2:
        raise ValueError(f"{patient['id']}: malformed patient-context conversation")

    result = []
    image_cursor = 0
    for index in range(1, len(messages), 2):
        question, answer = messages[index : index + 2]
        if question.get("from") != "human" or answer.get("from") != "gpt":
            raise ValueError(f"{patient['id']}: invalid roles at messages {index}/{index + 1}")
        image_count = question["value"].count("<image>")
        turn_images = patient["images"][image_cursor : image_cursor + image_count]
        if len(turn_images) != image_count:
            raise ValueError(f"{patient['id']}: image mismatch at turn {(index + 1) // 2}")
        result.append((question, answer, turn_images))
        image_cursor += image_count
    if image_cursor != len(patient["images"]):
        raise ValueError(f"{patient['id']}: unconsumed image references")
    return result


def make_turn_samples(patient: dict[str, Any], patient_staging: Path) -> list[dict[str, Any]]:
    history: list[tuple[str, str]] = []
    samples = []
    patient_system = patient["conversations"][0]["value"]
    for turn_index, (question, answer, turn_images) in enumerate(pairs(patient), start=1):
        sample_id = f"{patient['id']}__turn_{turn_index:02d}"
        sample = {
            "id": sample_id,
            "source_patient_id": patient["id"],
            "patient_uid": patient.get("patient_uid"),
            "patient_name": patient.get("patient_name"),
            "turn_index": turn_index,
            "context_turn_labels": patient.get("context_turn_labels", []),
            "conversations": [
                {"from": "system", "value": rollout_system_context(patient_system, history)},
                deepcopy(question),
                deepcopy(answer),
            ],
            "images": [link_media(patient_staging, OUTPUT, image) for image in turn_images],
            "num_images": len(turn_images),
            "num_qa_pairs": 1,
            "history_qa_pairs": len(history),
            "sft_source": patient.get("sft_source"),
        }
        samples.append(sample)
        history.append((question["value"], answer["value"]))
    return samples


def make_rollout_patient(patient: dict[str, Any], patient_staging: Path) -> dict[str, Any]:
    item = deepcopy(patient)
    item["images"] = [link_media(patient_staging, OUTPUT, image) for image in patient["images"]]
    return item


def validate_turn_samples(name: str, items: list[dict[str, Any]]) -> dict[str, int]:
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name}: duplicate turn sample IDs")
    patients = set()
    images = 0
    for item in items:
        messages = item["conversations"]
        if not messages or messages[0].get("from") != "system":
            raise ValueError(f"{name}/{item['id']}: missing system context")
        if "<image>" in messages[0].get("value", ""):
            raise ValueError(f"{name}/{item['id']}: system context has image placeholders")
        if (len(messages) - 1) % 2:
            raise ValueError(f"{name}/{item['id']}: invalid conversation length")
        for index, message in enumerate(messages[1:]):
            expected = "human" if index % 2 == 0 else "gpt"
            if message.get("from") != expected:
                raise ValueError(f"{name}/{item['id']}: expected {expected} at message {index + 1}")
        placeholders = sum(message["value"].count("<image>") for message in messages[1::2])
        if placeholders != len(item["images"]):
            raise ValueError(f"{name}/{item['id']}: image placeholder mismatch")
        if not all((OUTPUT / image).is_file() for image in item["images"]):
            raise FileNotFoundError(f"{name}/{item['id']}: missing generated media")
        if item["num_qa_pairs"] != 1:
            raise ValueError(f"{name}/{item['id']}: each turn sample must have one target")
        patients.add(item["source_patient_id"])
        images += len(item["images"])
    return {"turn_samples": len(items), "source_patients": len(patients), "image_references": images}


def main() -> None:
    patient_context, raw_staging = build_patient_context_staging()
    try:
        if OUTPUT.exists():
            shutil.rmtree(OUTPUT)
        OUTPUT.mkdir(parents=True)

        turn_splits: dict[str, list[dict[str, Any]]] = {}
        rollout_test: list[dict[str, Any]] = []
        patient_stats: dict[str, int] = {}
        for name in SPLITS:
            raw_items = json.loads((raw_staging / f"oralcmf_sft_{name}.json").read_text(encoding="utf-8"))
            patients = [patient_context.transform_item(item, raw_staging, PATIENT_STAGING)[0] for item in raw_items]
            turn_splits[name] = [sample for patient in patients for sample in make_turn_samples(patient, PATIENT_STAGING)]
            patient_stats[name] = len(patients)
            if name == "test":
                rollout_test = [make_rollout_patient(patient, PATIENT_STAGING) for patient in patients]

        expected_patients = {"train": 400, "validation": 50, "test": 50, "human_evaluation": 27}
        expected_turns = {"train": 5156, "validation": 649, "test": 650, "human_evaluation": 350}
        stats = {name: validate_turn_samples(name, items) for name, items in turn_splits.items()}
        if patient_stats != expected_patients or {name: value["turn_samples"] for name, value in stats.items()} != expected_turns:
            raise ValueError(f"Unexpected split sizes: patients={patient_stats}, turns={stats}")

        for name, items in turn_splits.items():
            write_json(OUTPUT / f"oralcmf_sft_{name}.json", items)
        write_json(OUTPUT / "oralcmf_sft_test_rollout.json", rollout_test)
        write_json(
            OUTPUT / "dataset_info.json",
            {
                f"oralcmf_sft_{name}": {
                    "file_name": f"oralcmf_sft_{name}.json",
                    "formatting": "sharegpt",
                    "columns": {"messages": "conversations", "images": "images"},
                    "tags": {
                        "role_tag": "from",
                        "content_tag": "value",
                        "user_tag": "human",
                        "assistant_tag": "gpt",
                        "system_tag": "system",
                    },
                }
                for name in SPLITS
            },
        )
        write_json(
            OUTPUT / "manifest.json",
            {
                "training_unit": "One supervised clinical-reasoning answer per sample.",
                "history_policy": "Prior non-context reference turns are serialized into the system clinical record; only the current answer is supervised.",
                "patient_context_policy": "Available Profile, TMJ, and ECT records are system context only.",
                "source_patients": patient_stats,
                "turn_sample_stats": stats,
                "rollout_test_file": "oralcmf_sft_test_rollout.json",
            },
        )
        print(json.dumps({"patients": patient_stats, "turn_samples": stats}, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(raw_staging, ignore_errors=True)
        shutil.rmtree(PATIENT_STAGING, ignore_errors=True)


if __name__ == "__main__":
    main()
