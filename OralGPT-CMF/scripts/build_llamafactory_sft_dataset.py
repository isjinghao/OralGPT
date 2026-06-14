#!/usr/bin/env python3
"""
Build a LLaMA-Factory ShareGPT-style SFT dataset from per-stage patient QA JSONs.

Each output sample represents one patient. QA pairs from all outputs/stage*
directories are concatenated in stage order and stored as a multi-turn dialogue:

[
  {
    "id": "group1__PATIENT",
    "images": ["/path/to/image1.png", "/path/to/image2.png"],
    "conversations": [
      {"from": "human", "value": "<image>\n<image>\n..."},
      {"from": "gpt", "value": "..."}
    ]
  }
]

Use this dataset with a LLaMA-Factory dataset_info.json entry similar to:

{
  "oral_cmf_sft": {
    "file_name": "oral_cmf_llamafactory_sft.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "images"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt"
    }
  }
}
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_OUTPUT_FILE = Path(__file__).resolve().parent / "oralgpt_cmf_llamafactory_sft_dataset.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-stage patient QA JSON files into one ShareGPT-style "
            "LLaMA-Factory SFT dataset."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing stage* output folders. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSON file. Default: {DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include QA entries whose status is not success if question and answer exist.",
    )
    parser.add_argument(
        "--write-dataset-info-snippet",
        type=Path,
        default=None,
        help="Optional path to write a dataset_info.json snippet for LLaMA-Factory.",
    )
    return parser.parse_args()


def stage_sort_key(stage_dir: Path) -> tuple[int, str]:
    match = re.match(r"stage(\d+)", stage_dir.name, flags=re.IGNORECASE)
    stage_number = int(match.group(1)) if match else 10**9
    return stage_number, stage_dir.name


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return textwrap.dedent(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return str(value).strip()


def iter_stage_dirs(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted(
        [path for path in input_dir.iterdir() if path.is_dir() and path.name.startswith("stage")],
        key=stage_sort_key,
    )


def normalize_images(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        path = normalize_text(value)
        return [path] if path else []
    if isinstance(value, list):
        images: list[str] = []
        for item in value:
            path = normalize_text(item)
            if path:
                images.append(path)
        return images
    return []


def add_image_placeholders(question: str, images: list[str]) -> str:
    if not images:
        return question
    placeholders = "\n".join("<image>" for _ in images)
    return f"{placeholders}\n{question}"


def extract_qa_pairs(data: dict[str, Any], include_failed: bool) -> list[dict[str, Any]]:
    modalities = data.get("Modalities")
    if not isinstance(modalities, dict):
        return []

    qa_pairs: list[dict[str, str]] = []
    for modality_name, entry in modalities.items():
        if not isinstance(entry, dict):
            continue
        status = normalize_text(entry.get("status")).lower()
        if not include_failed and status and status != "success":
            continue

        question = normalize_text(entry.get("question"))
        answer = normalize_text(entry.get("answer"))
        if not question or not answer:
            continue

        qa_pairs.append(
            {
                "modality": str(modality_name),
                "question": question,
                "answer": answer,
                "images": normalize_images(entry.get("input_images")),
            }
        )
    return qa_pairs


def load_patient_stage_records(input_dir: Path, include_failed: bool) -> dict[str, list[dict[str, Any]]]:
    records_by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for stage_dir in iter_stage_dirs(input_dir):
        for json_path in sorted(stage_dir.glob("*.json")):
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            patient_uid = data.get("patient_uid") or json_path.stem
            qa_pairs = extract_qa_pairs(data, include_failed=include_failed)
            if not qa_pairs:
                continue

            records_by_patient[str(patient_uid)].append(
                {
                    "stage_dir": stage_dir.name,
                    "stage_key": stage_sort_key(stage_dir),
                    "json_path": str(json_path),
                    "patient_name": data.get("patient_name"),
                    "group": data.get("group"),
                    "qa_pairs": qa_pairs,
                }
            )

    return records_by_patient


def build_conversation(patient_uid: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_records = sorted(records, key=lambda item: (item["stage_key"], item["json_path"]))
    conversations: list[dict[str, str]] = []
    images: list[str] = []
    stages: list[str] = []
    source_files: list[str] = []

    for record in sorted_records:
        stages.append(record["stage_dir"])
        source_files.append(record["json_path"])
        for qa in record["qa_pairs"]:
            qa_images = qa.get("images", [])
            images.extend(qa_images)
            conversations.append(
                {
                    "from": "human",
                    "value": add_image_placeholders(qa["question"], qa_images),
                }
            )
            conversations.append({"from": "gpt", "value": qa["answer"]})

    first_record = sorted_records[0]
    return {
        "id": patient_uid,
        "patient_uid": patient_uid,
        "patient_name": first_record.get("patient_name"),
        "group": first_record.get("group"),
        "stages": stages,
        "source_files": source_files,
        "images": images,
        "num_images": len(images),
        "num_qa_pairs": len(conversations) // 2,
        "conversations": conversations,
    }


def build_dataset(input_dir: Path, include_failed: bool) -> list[dict[str, Any]]:
    records_by_patient = load_patient_stage_records(input_dir, include_failed=include_failed)
    return [
        build_conversation(patient_uid, records_by_patient[patient_uid])
        for patient_uid in sorted(records_by_patient)
    ]


def write_dataset_info_snippet(output_path: Path, dataset_file_name: str) -> None:
    snippet = {
        "oral_cmf_sft": {
            "file_name": dataset_file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(snippet, f, ensure_ascii=False, indent=2)
        f.write("\n")


def count_image_qa_pairs(item: dict[str, Any]) -> int:
    return sum(
        1
        for turn in item.get("conversations", [])
        if turn.get("from") == "human" and "<image>" in turn.get("value", "")
    )


def print_dataset_stats(dataset: list[dict[str, Any]], output_file: Path) -> None:
    total_patients = len(dataset)
    total_qa_pairs = sum(item.get("num_qa_pairs", 0) for item in dataset)
    total_images = sum(item.get("num_images", len(item.get("images", []))) for item in dataset)
    total_image_qa_pairs = sum(count_image_qa_pairs(item) for item in dataset)
    total_text_only_qa_pairs = total_qa_pairs - total_image_qa_pairs
    patients_with_images = sum(1 for item in dataset if item.get("images"))
    avg_qa_per_patient = total_qa_pairs / total_patients if total_patients else 0
    avg_images_per_patient = total_images / total_patients if total_patients else 0
    avg_images_per_image_patient = total_images / patients_with_images if patients_with_images else 0
    max_images_per_patient = max((len(item.get("images", [])) for item in dataset), default=0)
    max_qa_per_patient = max((item.get("num_qa_pairs", 0) for item in dataset), default=0)
    placeholder_total = sum(
        turn.get("value", "").count("<image>")
        for item in dataset
        for turn in item.get("conversations", [])
    )

    print(f"Wrote {total_patients} patient conversations to {output_file}")
    print("Dataset statistics:")
    print(f"  Total patients: {total_patients}")
    print(f"  Total QA pairs: {total_qa_pairs}")
    print(f"  Total images: {total_images}")
    print(f"  Patients with images: {patients_with_images}")
    print(f"  QA pairs with images: {total_image_qa_pairs}")
    print(f"  Text-only QA pairs: {total_text_only_qa_pairs}")
    print(f"  Average QA pairs per patient: {avg_qa_per_patient:.2f}")
    print(f"  Average images per patient: {avg_images_per_patient:.2f}")
    print(f"  Average images per patient with images: {avg_images_per_image_patient:.2f}")
    print(f"  Max QA pairs per patient: {max_qa_per_patient}")
    print(f"  Max images per patient: {max_images_per_patient}")
    print(f"  Image placeholders: {placeholder_total}")
    print(f"  Image paths match placeholders: {total_images == placeholder_total}")


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args.input_dir, include_failed=args.include_failed)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if args.write_dataset_info_snippet:
        write_dataset_info_snippet(
            args.write_dataset_info_snippet,
            dataset_file_name=args.output_file.name,
        )

    print_dataset_stats(dataset, args.output_file)
    if args.write_dataset_info_snippet:
        print(f"Wrote dataset_info snippet to {args.write_dataset_info_snippet}")


if __name__ == "__main__":
    main()
