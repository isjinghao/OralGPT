#!/usr/bin/env python3
"""Export MRI T1<->T2 test parquet shards into OralGPT-X-Bench metadata + PNG files."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from path_utils import bench_root, resolve_path

TASK_NAMES = {
    "mri_t1_to_t2": "mri_t1_to_t2",
    "mri_t2_to_t1": "mri_t2_to_t1",
}


def pair_id_to_sample_id(pair_id: str) -> str:
    return pair_id.replace("::", "_").replace("/", "_")


def png_bytes_to_file(png_bytes: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(png_bytes)) as image:
        image.save(path, format="PNG")


def export_task(
    parquet_path: Path,
    task_type: str,
    bench_data_root: Path,
    samples: list[dict],
    *,
    max_per_task: int | None,
    max_total: int | None,
) -> None:
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if max_per_task is not None:
        rows = rows[:max_per_task]
    for row in tqdm(rows, desc=task_type):
        if max_total is not None and len(samples) >= max_total:
            return
        pair_id = row["pair_id"]
        instruction = row["instruction_list"][0][0]
        sample_id = pair_id_to_sample_id(pair_id)

        source_rel = Path("source") / task_type / f"{sample_id}.png"
        target_rel = Path("target") / task_type / f"{sample_id}.png"
        png_bytes_to_file(row["image_list"][0], bench_data_root / source_rel)
        png_bytes_to_file(row["image_list"][1], bench_data_root / target_rel)

        samples.append(
            {
                "id": sample_id,
                "benchmark": "mri",
                "task_family": "edit_translation",
                "task_type": task_type,
                "split": "test",
                "source": {"image_path": source_rel.as_posix()},
                "target": {"image_path": target_rel.as_posix(), "role": "pixel_gt"},
                "instruction": instruction,
                "metadata": {
                    "pair_id": pair_id,
                    "cohort": row.get("cohort") or row.get("source_cohort"),
                    "volume_id": row.get("volume_id"),
                    "source_modality": row.get("source_modality"),
                    "target_modality": row.get("target_modality"),
                    "direction": row.get("direction"),
                    "source_slice_index": row.get("source_slice_index"),
                    "target_slice_index": row.get("target_slice_index"),
                },
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet-root",
        type=Path,
        default=Path("/data/OralGPT/OralGPT-X/dataset_MRI_T1_T2/test"),
        help="Root containing mri_t1_to_t2/ and mri_t2_to_t1/ parquet shards",
    )
    parser.add_argument(
        "--bench-data-root",
        type=Path,
        default=bench_root() / "benchmark_data" / "mri",
        help="Directory for exported source/target PNG files",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=bench_root() / "benchmark" / "mri" / "metadata.test.json",
    )
    parser.add_argument("--tasks", nargs="*", default=list(TASK_NAMES.keys()))
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=None,
        help="If set, export at most N samples per task.",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=None,
        help="If set, stop after exporting this many samples overall.",
    )
    args = parser.parse_args()

    parquet_root = resolve_path(args.parquet_root)
    bench_data_root = resolve_path(args.bench_data_root)
    output_metadata = resolve_path(args.output_metadata, bench_root())

    samples: list[dict] = []
    for task_dir_name in args.tasks:
        if args.max_total is not None and len(samples) >= args.max_total:
            break
        parquet_path = parquet_root / task_dir_name / "part-00000.parquet"
        if not parquet_path.is_file():
            raise FileNotFoundError(parquet_path)
        export_task(
            parquet_path,
            task_dir_name,
            bench_data_root,
            samples,
            max_per_task=args.max_per_task,
            max_total=args.max_total,
        )

    payload = {
        "benchmark": "mri",
        "version": "v1.0",
        "split": "test",
        "num_samples": len(samples),
        "bench_data_root_hint": str(bench_data_root),
        "samples": samples,
    }
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(samples)} samples")
    print(f"Images: {bench_data_root}")
    print(f"Metadata: {output_metadata}")


if __name__ == "__main__":
    main()
