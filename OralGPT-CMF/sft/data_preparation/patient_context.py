#!/usr/bin/env python3
"""Convert OralCMF SFT data into patient-context supervision data."""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
OUTPUT = ROOT / "data/patient_context"
STAGING = ROOT / ".sft_data_patient_context_staging"
SPLITS = ("train", "validation", "test", "human_evaluation")


def context_label(question: str) -> str | None:
    text = re.sub(r"(?:<image>\s*)+", "", question).lower()
    if "single-photon emission" in text or "(ect)" in text:
        return "ECT"
    if "temporomandibular joint" in text or "(tmj)" in text:
        return "TMJ"
    if any(
        phrase in text
        for phrase in (
            "basic information",
            "primary concern",
            "past medical",
            "medical or surgical history",
            "chief complaint",
        )
    ):
        return "PROFILE"
    return None


def clean_context_question(question: str) -> str:
    return re.sub(r"\s*<image>\s*", " ", question).strip()


def patient_context(records: list[tuple[str, str, str]]) -> str:
    lines = [
        "Patient-specific clinical context. Treat the following as prior clinical record.",
        "Use it when relevant, and do not invent facts beyond it.",
    ]
    for label, question, answer in records:
        lines.extend(("", f"[{label}]", f"Question: {clean_context_question(question)}", f"Recorded answer: {answer.strip()}"))
    return "\n".join(lines)


def link_media(source_dataset: Path, destination: Path, source_relative: str) -> str:
    source_path = (source_dataset / source_relative).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source media: {source_path}")
    destination_path = destination / source_relative
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() or destination_path.is_symlink():
        destination_path.unlink()
    destination_path.symlink_to(source_path)
    return source_relative


def transform_item(raw_item: dict[str, Any], source_dataset: Path, destination: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    item = deepcopy(raw_item)
    messages = item["conversations"]
    if len(messages) % 2:
        raise ValueError(f"{item['id']}: odd conversation length")

    retained: list[dict[str, str]] = []
    context_records: list[tuple[str, str, str]] = []
    images: list[str] = []
    image_cursor = 0
    excluded_images = 0
    for index in range(0, len(messages), 2):
        question, answer = messages[index : index + 2]
        if question.get("from") != "human" or answer.get("from") != "gpt":
            raise ValueError(f"{item['id']}: invalid roles at messages {index}/{index + 1}")
        image_count = question.get("value", "").count("<image>")
        turn_images = raw_item.get("images", [])[image_cursor : image_cursor + image_count]
        if len(turn_images) != image_count:
            raise ValueError(f"{item['id']}: image references do not match turn {index // 2 + 1}")
        image_cursor += image_count
        label = context_label(question["value"])
        if label:
            context_records.append((label, question["value"], answer["value"]))
            excluded_images += image_count
            continue
        retained.extend((question, answer))
        images.extend(link_media(source_dataset, destination, image) for image in turn_images)

    if image_cursor != len(raw_item.get("images", [])):
        raise ValueError(f"{item['id']}: unconsumed image references")
    if not retained:
        raise ValueError(f"{item['id']}: missing supervised turns")

    item["conversations"] = [{"from": "system", "value": patient_context(context_records)}, *retained]
    item["images"] = images
    item["num_images"] = len(images)
    item["num_qa_pairs"] = len(retained) // 2
    item["context_turn_labels"] = [label for label, _, _ in context_records]
    item["context_turn_count"] = len(context_records)
    item["supervised_qa_pairs"] = len(retained) // 2
    audit = {
        "id": item["id"],
        "context_turn_labels": item["context_turn_labels"],
        "context_turn_count": item["context_turn_count"],
        "supervised_qa_pairs": item["supervised_qa_pairs"],
        "excluded_context_images": excluded_images,
        "retained_images": len(images),
    }
    return item, audit


def validate_split(name: str, items: list[dict[str, Any]], destination: Path) -> dict[str, Any]:
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name}: duplicate IDs")
    context_counts = Counter()
    supervised_pairs = 0
    images = 0
    for item in items:
        messages = item["conversations"]
        if not messages or messages[0].get("from") != "system" or not messages[0].get("value", "").strip():
            raise ValueError(f"{name}/{item['id']}: missing system context")
        if "<image>" in messages[0]["value"]:
            raise ValueError(f"{name}/{item['id']}: system context contains image placeholders")
        if (len(messages) - 1) % 2:
            raise ValueError(f"{name}/{item['id']}: invalid remaining message count")
        for index, message in enumerate(messages[1:]):
            expected = "human" if index % 2 == 0 else "gpt"
            if message.get("from") != expected:
                raise ValueError(f"{name}/{item['id']}: expected {expected} at message {index + 1}")
        placeholders = sum(message["value"].count("<image>") for message in messages[1::2])
        if placeholders != len(item["images"]):
            raise ValueError(f"{name}/{item['id']}: image mismatch {placeholders} != {len(item['images'])}")
        if not all((destination / image).is_file() for image in item["images"]):
            raise FileNotFoundError(f"{name}/{item['id']}: missing generated media")
        if item["num_qa_pairs"] != (len(messages) - 1) // 2:
            raise ValueError(f"{name}/{item['id']}: incorrect supervised pair count")
        context_counts.update(item["context_turn_labels"])
        supervised_pairs += item["num_qa_pairs"]
        images += len(item["images"])
    return {"items": len(items), "supervised_qa_pairs": supervised_pairs, "images": images, "context_turns": dict(context_counts)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_staging_dataset() -> Path:
    from sft.data_preparation import split_dataset as raw

    if STAGING.exists():
        shutil.rmtree(STAGING)
    raw.build_dataset(STAGING)
    return STAGING


def main() -> None:
    source = build_staging_dataset()
    try:
        if OUTPUT.exists():
            shutil.rmtree(OUTPUT)
        OUTPUT.mkdir(parents=True)

        splits: dict[str, list[dict[str, Any]]] = {}
        audit: dict[str, list[dict[str, Any]]] = {}
        for name in SPLITS:
            source_path = source / f"oralcmf_sft_{name}.json"
            raw_items = json.loads(source_path.read_text(encoding="utf-8"))
            transformed = [transform_item(item, source, OUTPUT) for item in raw_items]
            splits[name] = [item for item, _ in transformed]
            audit[name] = [item_audit for _, item_audit in transformed]

        expected = {"train": 400, "validation": 50, "test": 50, "human_evaluation": 27}
        stats = {name: validate_split(name, items, OUTPUT) for name, items in splits.items()}
        if {name: value["items"] for name, value in stats.items()} != expected:
            raise ValueError(f"Unexpected split sizes: {stats}")

        for name, items in splits.items():
            write_json(OUTPUT / f"oralcmf_sft_{name}.json", items)
        dataset_info = {
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
        }
        write_json(OUTPUT / "dataset_info.json", dataset_info)
        write_json(
            OUTPUT / "manifest.json",
            {
                "source_dataset": "Regenerated from Final_VQADataset source files.",
                "context_policy": "Profile, TMJ, and ECT question-answer pairs are system context, not SFT targets.",
                "stats": stats,
                "ids": {name: [item["id"] for item in items] for name, items in splits.items()},
                "audit": audit,
            },
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(source, ignore_errors=True)


if __name__ == "__main__":
    main()
