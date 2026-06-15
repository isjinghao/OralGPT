#!/usr/bin/env python3
"""Merge cohort-level MRI T1/T2 BAGEL parquet datasets into one MRI subset.

The merged rows preserve provenance columns such as source_cohort,
source_dataset_root, source_parquet_path, and original_pair_id.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--cohort",
        action="append",
        nargs=2,
        metavar=("NAME", "DATASET_ROOT"),
        required=True,
        help="Source cohort name and its BAGEL dataset root. Repeat for each cohort.",
    )
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--row-group-size", type=int, default=256)
    parser.add_argument("--dataset-slug", default="mri_t1_t2_all")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def iter_rows(paths: list[Path]):
    for source_path in paths:
        pf = pq.ParquetFile(source_path)
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                yield source_path, row


def normalize_row(
    row: dict,
    *,
    cohort: str,
    cohort_root: Path,
    split: str,
    direction: str,
    source_parquet: Path,
) -> dict:
    out_row = dict(row)
    original_pair_id = out_row.get("pair_id")
    out_row["source_cohort"] = cohort
    out_row["source_subset"] = split
    out_row["source_direction"] = direction
    out_row["source_dataset_root"] = str(cohort_root)
    out_row["source_parquet_path"] = str(source_parquet)
    out_row["original_pair_id"] = original_pair_id
    out_row["pair_id"] = f"{cohort}::{original_pair_id}"
    return out_row


def write_shard(rows: list[dict], path: Path, row_group_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, row_group_size=row_group_size)


def write_bagel_snippets(
    output_root: Path,
    dataset_slug: str,
    counts: dict[str, dict[str, int]],
    parquet_outputs: dict[str, list[Path]],
) -> None:
    directions = {"t1_to_t2": "mri_t1_to_t2", "t2_to_t1": "mri_t2_to_t1"}
    info_lines = ["# Add these entries under DATASET_INFO['unified_edit']."]
    dataset_names: list[str] = []
    for direction, dataset_dir in directions.items():
        ds_name = f"{dataset_slug}_{direction}_train"
        dataset_names.append(ds_name)
        train_dir = output_root / "train" / dataset_dir
        info_path = output_root / "parquet_info" / f"train_{direction}.json"
        info_lines.append(
            f"'{ds_name}': {{\n"
            f"    'data_dir': '{train_dir.resolve()}',\n"
            f"    'num_files': {len(parquet_outputs[f'train_{direction}'])},\n"
            f"    'num_total_samples': {counts['train'][direction]},\n"
            f"    'parquet_info_path': '{info_path.resolve()}',\n"
            f"}},"
        )
    (output_root / "bagel_dataset_info_snippet.py").write_text(
        "\n".join(info_lines) + "\n", encoding="utf-8"
    )

    names_yaml = "\n".join(f"    - {name}" for name in dataset_names)
    config_yaml = f"""# Merged MRI T1/T2 subset
unified_edit:
  dataset_names:
{names_yaml}
  image_transform_args:
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512
  vit_image_transform_args:
    image_stride: 14
    max_image_size: 518
    min_image_size: 224
  is_mandatory: false
  num_used_data:
    - 1
    - 1
  weight: 1
"""
    (output_root / "bagel_example_config_snippet.yaml").write_text(
        config_yaml, encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_root}. Use --overwrite to allow it.")

    cohorts = [(name, Path(root).expanduser().resolve()) for name, root in args.cohort]
    directions = {"t1_to_t2": "mri_t1_to_t2", "t2_to_t1": "mri_t2_to_t1"}
    splits = ["train", "test"]

    inputs: dict[tuple[str, str, str], list[Path]] = {}
    for cohort, root in cohorts:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for split in splits:
            for direction, dataset_dir in directions.items():
                paths = sorted((root / split / dataset_dir).glob("part-*.parquet"))
                if not paths:
                    raise FileNotFoundError(root / split / dataset_dir)
                inputs[(cohort, split, direction)] = paths

    counts = {split: {direction: 0 for direction in directions} for split in splits}
    cohort_counts = {
        cohort: {split: {direction: 0 for direction in directions} for split in splits}
        for cohort, _ in cohorts
    }
    parquet_outputs: dict[str, list[Path]] = defaultdict(list)
    source_manifests: dict[str, object] = {}

    for cohort, root in cohorts:
        manifest_path = root / "split_manifest.json"
        if manifest_path.is_file():
            source_manifests[cohort] = json.loads(manifest_path.read_text(encoding="utf-8"))

    for split in splits:
        for direction, dataset_dir in directions.items():
            shard_rows: list[dict] = []
            part_idx = 0
            for cohort, root in cohorts:
                for source_path, row in iter_rows(inputs[(cohort, split, direction)]):
                    shard_rows.append(
                        normalize_row(
                            row,
                            cohort=cohort,
                            cohort_root=root,
                            split=split,
                            direction=direction,
                            source_parquet=source_path,
                        )
                    )
                    counts[split][direction] += 1
                    cohort_counts[cohort][split][direction] += 1
                    if len(shard_rows) >= args.shard_size:
                        out_path = output_root / split / dataset_dir / f"part-{part_idx:05d}.parquet"
                        write_shard(shard_rows, out_path, args.row_group_size)
                        parquet_outputs[f"{split}_{direction}"].append(out_path)
                        print(f"wrote {out_path} rows={len(shard_rows)}", flush=True)
                        part_idx += 1
                        shard_rows = []
            if shard_rows:
                out_path = output_root / split / dataset_dir / f"part-{part_idx:05d}.parquet"
                write_shard(shard_rows, out_path, args.row_group_size)
                parquet_outputs[f"{split}_{direction}"].append(out_path)
                print(f"wrote {out_path} rows={len(shard_rows)}", flush=True)

    info_root = output_root / "parquet_info"
    info_root.mkdir(parents=True, exist_ok=True)
    for split in splits:
        for direction in directions:
            paths = parquet_outputs[f"{split}_{direction}"]
            info = {
                str(path.resolve()): {"num_row_groups": pq.ParquetFile(path).num_row_groups}
                for path in paths
            }
            (info_root / f"{split}_{direction}.json").write_text(
                json.dumps(info, indent=2), encoding="utf-8"
            )

    manifest = {
        "name": args.dataset_slug,
        "description": "Merged BAGEL unified_edit MRI modality translation subset.",
        "source_cohorts": [cohort for cohort, _ in cohorts],
        "source_dataset_roots": {cohort: str(root) for cohort, root in cohorts},
        "provenance_columns": [
            "source_cohort",
            "source_subset",
            "source_direction",
            "source_dataset_root",
            "source_parquet_path",
            "original_pair_id",
            "cohort",
            "volume_id",
            "pair_id",
        ],
        "shard_size": args.shard_size,
        "row_group_size": args.row_group_size,
        "counts": counts,
        "counts_by_source_cohort": cohort_counts,
        "source_split_manifests": source_manifests,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    write_bagel_snippets(output_root, args.dataset_slug, counts, parquet_outputs)
    summary = {
        "output_root": str(output_root),
        "counts": counts,
        "counts_by_source_cohort": cohort_counts,
        "bagel_dataset_names": [
            f"{args.dataset_slug}_t1_to_t2_train",
            f"{args.dataset_slug}_t2_to_t1_train",
        ],
        "train_total": sum(counts["train"].values()),
        "test_total": sum(counts["test"].values()),
        "total": sum(sum(v.values()) for v in counts.values()),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
