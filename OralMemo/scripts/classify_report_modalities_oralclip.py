#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

MODALITY_COLUMNS = [
    "intraoral",
    "periapical_xray",
    "panoramic_xray",
    "CT",
    "histopathology",
    "cytology",
    "cephalometric_xray",
    "speech_mri",
    "others",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_path(root: Path, raw_path: str) -> tuple[str, Path]:
    path = Path(raw_path)
    abs_path = path if path.is_absolute() else root / path
    try:
        rel_path = abs_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel_path = abs_path.as_posix()
    return rel_path, abs_path


def collect_standard_images(root: Path, limit_cases: int | None = None) -> tuple[list[dict], list[dict]]:
    report_root = root / "outputs" / "report"
    trajectories = sorted(report_root.glob("CR*/trajectories/standard_trajectory.json"))
    if limit_cases is not None:
        trajectories = trajectories[:limit_cases]

    items: dict[str, dict] = {}
    missing: list[dict] = []

    for trajectory_path in trajectories:
        case_id = trajectory_path.parents[1].name
        trajectory = read_json(trajectory_path)
        for stage in trajectory.get("stages", []):
            stage_id = stage.get("stage_id", "")
            stage_type = stage.get("stage_type", "")
            refs = list(stage.get("image_paths", []))
            for qa in stage.get("qa_pairs", []):
                refs.extend(qa.get("image_paths", []))
            for raw_path in refs:
                rel_path, abs_path = normalize_path(root, raw_path)
                if not abs_path.is_file():
                    missing.append({"case_id": case_id, "image_path": rel_path, "stage_id": stage_id})
                    continue
                item = items.setdefault(
                    rel_path,
                    {
                        "case_id": case_id,
                        "image_path": rel_path,
                        "abs_path": abs_path,
                        "stages": set(),
                        "stage_types": set(),
                        "num_references": 0,
                    },
                )
                item["stages"].add(stage_id)
                item["stage_types"].add(stage_type)
                item["num_references"] += 1

    records = []
    for item in items.values():
        records.append(
            {
                **item,
                "stages": sorted(item["stages"]),
                "stage_types": sorted(item["stage_types"]),
            }
        )
    records.sort(key=lambda x: x["image_path"])
    return records, missing


def write_outputs(records: list[dict], missing: list[dict], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    jsonl_path = output_prefix.with_suffix(".jsonl")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")

    fieldnames = [
        "case_id",
        "image_path",
        "predicted_modality",
        "confidence",
        "top3",
        "stages",
        "stage_types",
        "num_references",
        *[f"prob_{m}" for m in MODALITY_COLUMNS],
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in fieldnames}
            row["top3"] = json.dumps(record.get("top3", []), ensure_ascii=False)
            row["stages"] = ";".join(record.get("stages", []))
            row["stage_types"] = ";".join(record.get("stage_types", []))
            writer.writerow(row)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    modality_counts = Counter(record["predicted_modality"] for record in records)
    case_counts = defaultdict(Counter)
    for record in records:
        case_counts[record["case_id"]][record["predicted_modality"]] += 1

    summary = {
        "total_unique_images": len(records),
        "missing_images": missing,
        "modality_counts": dict(sorted(modality_counts.items())),
        "case_modality_counts": {case: dict(counts) for case, counts in sorted(case_counts.items())},
        "outputs": {
            "csv": csv_path.as_posix(),
            "jsonl": jsonl_path.as_posix(),
            "summary": summary_path.as_posix(),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved_csv={csv_path}")
    print(f"saved_jsonl={jsonl_path}")
    print(f"saved_summary={summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify OralMemo report images with OralCLIP modality head.")
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/OralGPT/OralMemo"))
    parser.add_argument("--oralclip-dir", type=Path, default=Path("/root/autodl-tmp/OralGPT/OralDetect/OralCLIP"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--text-tower", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, default=Path("outputs/report/oralclip_modality_predictions"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    oralclip_dir = args.oralclip_dir.resolve()
    checkpoint = args.checkpoint or oralclip_dir / "weights" / "OralCLIP" / "oralclip.pt"
    text_tower = args.text_tower or oralclip_dir / "weights" / "OralCLIP" / "oralbert"
    output_prefix = args.output_prefix
    if not output_prefix.is_absolute():
        output_prefix = root / output_prefix

    records, missing = collect_standard_images(root, args.limit_cases)
    if args.limit_images is not None:
        records = records[: args.limit_images]
    print(f"unique_standard_images={len(records)}")
    print(f"missing_images={len(missing)}")
    if args.dry_run:
        return 0
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} referenced images; first={missing[0]}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not text_tower.is_dir():
        raise FileNotFoundError(f"Missing text tower: {text_tower}")

    sys.path.insert(0, str(oralclip_dir))
    from infer import encode_images
    from model import MODALITIES, load_oralclip

    if list(MODALITIES) != MODALITY_COLUMNS:
        raise ValueError(f"Unexpected modality order: {MODALITIES}")

    model = load_oralclip(str(checkpoint), str(text_tower), device=args.device)
    _, logits = encode_images(
        model,
        [str(record["abs_path"]) for record in records],
        device=args.device,
        batch_size=args.batch_size,
        return_modality=True,
    )
    probs = logits.softmax(dim=-1)

    output_records = []
    for record, prob in zip(records, probs):
        top = torch.topk(prob, k=3)
        top3 = [
            {"modality": MODALITIES[idx], "prob": round(float(value), 6)}
            for value, idx in zip(top.values.tolist(), top.indices.tolist())
        ]
        row = {
            "case_id": record["case_id"],
            "image_path": record["image_path"],
            "predicted_modality": top3[0]["modality"],
            "confidence": top3[0]["prob"],
            "top3": top3,
            "stages": record["stages"],
            "stage_types": record["stage_types"],
            "num_references": record["num_references"],
        }
        for modality, value in zip(MODALITIES, prob.tolist()):
            row[f"prob_{modality}"] = round(float(value), 6)
        output_records.append(row)

    write_outputs(output_records, missing, output_prefix)
    print("modality_counts=" + json.dumps(Counter(r["predicted_modality"] for r in output_records), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
