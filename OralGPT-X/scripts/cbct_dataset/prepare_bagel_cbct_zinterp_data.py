#!/usr/bin/env python3
"""Prepare BAGEL parquet shards for CBCT z-axis interpolation (131 -> 333).

Each sample uses two adjacent low-dose slices (131) to predict the physically
aligned middle standard-dose slice (333) between them.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import random
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pydicom
from PIL import Image

PREPROCESS_PATH = Path(__file__).resolve().with_name("cbct_preprocess.py")
PREPROCESS_SPEC = importlib.util.spec_from_file_location("cbct_preprocess", PREPROCESS_PATH)
if PREPROCESS_SPEC is None or PREPROCESS_SPEC.loader is None:
    raise ImportError(f"Cannot load preprocessing helpers from {PREPROCESS_PATH}")
PREPROCESS_MODULE = importlib.util.module_from_spec(PREPROCESS_SPEC)
PREPROCESS_SPEC.loader.exec_module(PREPROCESS_MODULE)
dicom_to_preprocessed_uint8 = PREPROCESS_MODULE.dicom_to_preprocessed_uint8


DEFAULT_OUTPUT = Path.cwd() / "dataset_Low-Dose_to_Standard_CBCT"
DEFAULT_SPLIT_MANIFEST = DEFAULT_OUTPUT / "split_manifest.json"
PREPROCESS_SLICE_START = 50
PREPROCESS_SLICE_STOP = 240

INSTRUCTION = (
    "Task: CBCT z-axis super-resolution via slice interpolation. "
    "Input: two adjacent 131 mGy.cm^2 low-dose, low-resolution axial slices. "
    "Target: the 333 mGy.cm^2 standard-dose, normal-resolution axial slice "
    "located at the physical midpoint between the two input slices. "
    "Instruction: Interpolate anatomically consistent intensity between the "
    "adjacent low-dose slices, recover fine osseous detail at standard-dose "
    "resolution, and preserve exact anatomy without hallucination."
)


def dicom_to_uint8_png_bytes(dcm_path: Path) -> bytes:
    ds = pydicom.dcmread(str(dcm_path), force=True)
    image_u8 = dicom_to_preprocessed_uint8(ds)

    buf = io.BytesIO()
    Image.fromarray(image_u8).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def slice_index(path: Path) -> int | None:
    match = re.search(r"_(\d+)\.dcm$", path.name)
    return int(match.group(1)) if match else None


def indexed_dicoms(folder: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in folder.glob("*.dcm"):
        idx = slice_index(path)
        if idx is not None:
            result[idx] = path
    return result


def list_volume_bases(input_root: Path) -> list[str]:
    bases = []
    folder_names = {path.name for path in input_root.iterdir() if path.is_dir()}
    for name in sorted(folder_names):
        if not name.endswith("_131"):
            continue
        base = name.rsplit("_", 1)[0]
        if f"{base}_333" in folder_names:
            bases.append(base)
    return bases


def load_split_manifest(manifest_path: Path) -> dict[str, list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_131 = manifest["datasets"]["131"]
    return {"train": split_131["train"], "test": split_131["test"]}


def split_bases_random(
    bases: list[str],
    train_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    shuffled = list(bases)
    random.Random(seed).shuffle(shuffled)
    train_count = round(len(shuffled) * train_ratio)
    return sorted(shuffled[:train_count]), sorted(shuffled[train_count:])


def is_kept_after_preprocess(slice_idx: int) -> bool:
    return PREPROCESS_SLICE_START <= slice_idx <= PREPROCESS_SLICE_STOP


def preprocess_slice_index(original_slice_idx: int) -> int:
    if not is_kept_after_preprocess(original_slice_idx):
        raise ValueError(f"Slice {original_slice_idx} is outside the preprocessed range")
    return original_slice_idx - PREPROCESS_SLICE_START + 1


def pair_similarity(source_a_png: bytes, source_b_png: bytes, target_png: bytes) -> float:
    source_a = Image.open(io.BytesIO(source_a_png)).convert("L")
    source_b = Image.open(io.BytesIO(source_b_png)).convert("L")
    target = Image.open(io.BytesIO(target_png)).convert("L")

    if source_a.size != target.size:
        source_a = source_a.resize(target.size, Image.Resampling.LANCZOS)
    if source_b.size != target.size:
        source_b = source_b.resize(target.size, Image.Resampling.LANCZOS)

    source_arr = (
        np.asarray(source_a, dtype=np.float32)
        + np.asarray(source_b, dtype=np.float32)
    ) / 2.0
    target_arr = np.asarray(target, dtype=np.float32)

    source_arr = source_arr.ravel()
    target_arr = target_arr.ravel()
    source_arr -= float(source_arr.mean())
    target_arr -= float(target_arr.mean())

    denominator = float(np.linalg.norm(source_arr) * np.linalg.norm(target_arr))
    if denominator == 0:
        return 0.0
    return float(np.dot(source_arr, target_arr) / denominator)


def volume_similarity(input_root: Path, base: str) -> tuple[float, int]:
    rows = build_rows_for_volume(input_root, base)
    if not rows:
        return float("-inf"), 0
    scores = [
        pair_similarity(
            row["image_list"][0],
            row["image_list"][1],
            row["image_list"][2],
        )
        for row in rows
    ]
    return float(sum(scores) / len(scores)), len(scores)


def split_bases_by_similarity(
    input_root: Path,
    bases: list[str],
    train_ratio: float,
) -> tuple[list[str], list[str], list[dict[str, object]]]:
    test_count = len(bases) - round(len(bases) * train_ratio)
    if test_count <= 0:
        return sorted(bases), [], []

    report_rows = []
    for idx, base in enumerate(bases, start=1):
        avg_similarity, num_pairs = volume_similarity(input_root, base)
        report_rows.append(
            {
                "volume_id": base,
                "avg_similarity": avg_similarity,
                "num_pairs": num_pairs,
            }
        )
        print(
            f"similarity 131_zinterp {idx}/{len(bases)} {base}: "
            f"avg={avg_similarity:.6f} pairs={num_pairs}",
            flush=True,
        )

    ranked = sorted(
        report_rows,
        key=lambda item: (item["avg_similarity"], item["volume_id"]),
        reverse=True,
    )
    test_bases = sorted(str(item["volume_id"]) for item in ranked[:test_count])
    test_set = set(test_bases)
    train_bases = sorted(base for base in bases if base not in test_set)

    for item in report_rows:
        item["split"] = "test" if item["volume_id"] in test_set else "train"

    return train_bases, test_bases, sorted(report_rows, key=lambda item: str(item["volume_id"]))


def build_rows_for_volume(input_root: Path, base: str) -> list[dict[str, object]]:
    """Build z-interpolation rows for one 131/333 volume pair."""
    low_dir = input_root / f"{base}_131"
    target_dir = input_root / f"{base}_333"
    low_slices = indexed_dicoms(low_dir)
    target_slices = indexed_dicoms(target_dir)

    rows: list[dict[str, object]] = []
    # Adjacent original 131 slices k and k+1; target is original 333 slice 2*k+1.
    for low_idx in range(1, 125):
        next_idx = low_idx + 1
        target_idx = 2 * low_idx + 1
        if (
            not is_kept_after_preprocess(low_idx)
            or not is_kept_after_preprocess(next_idx)
            or not is_kept_after_preprocess(target_idx)
        ):
            continue

        low_path = low_slices.get(low_idx)
        next_path = low_slices.get(next_idx)
        target_path = target_slices.get(target_idx)
        if low_path is None or next_path is None or target_path is None:
            continue

        source_preprocessed_idx = preprocess_slice_index(low_idx)
        source_preprocessed_idx_next = preprocess_slice_index(next_idx)
        target_preprocessed_idx = preprocess_slice_index(target_idx)

        rows.append(
            {
                "image_list": [
                    dicom_to_uint8_png_bytes(low_path),
                    dicom_to_uint8_png_bytes(next_path),
                    dicom_to_uint8_png_bytes(target_path),
                ],
                "instruction_list": [[INSTRUCTION]],
                "source_dose": "131",
                "target_dose": "333",
                "volume_id": base,
                "source_slice_index": source_preprocessed_idx,
                "source_slice_index_next": source_preprocessed_idx_next,
                "target_slice_index": target_preprocessed_idx,
                "source_original_slice_index": low_idx,
                "source_original_slice_index_next": next_idx,
                "target_original_slice_index": target_idx,
                "pair_id": (
                    f"{base}_131_{source_preprocessed_idx:03d}_"
                    f"{source_preprocessed_idx_next:03d}_to_333_"
                    f"{target_preprocessed_idx:03d}_orig_{low_idx:03d}_"
                    f"{next_idx:03d}_to_{target_idx:03d}"
                ),
            }
        )
    return rows


def write_parquet(rows: list[dict[str, object]], output_path: Path, row_group_size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path, row_group_size=row_group_size)


def write_parquet_info(parquet_paths: list[Path], output_path: Path) -> None:
    info = {
        str(path.resolve()): {"num_row_groups": pq.ParquetFile(path).num_row_groups}
        for path in parquet_paths
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def write_split_manifest(
    output_root: Path,
    split_map: dict[str, list[str]],
    counts: dict[str, int],
    split_strategy: str,
) -> None:
    manifest = {
        "split_policy": f"volume-level {split_strategy} split",
        "pairing_rules": {
            "131_zinterp_to_333": (
                "source slices k,k+1 (131) -> target slice 2*k+1 (333), "
                "physical midpoint between adjacent low-dose slices"
            ),
        },
        "datasets": {"131_zinterp": split_map},
        "counts": counts,
    }
    (output_root / "split_manifest_131_zinterp_to_333.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def write_bagel_snippets(output_root: Path, counts: dict[str, int]) -> None:
    train_dir = output_root / "train" / "cbct_131_zinterp_to_333"
    info_path = output_root / "parquet_info" / "train_cbct_131_zinterp_to_333.json"
    dataset_info = f"""# Add under DATASET_INFO['unified_edit'].
'cbct_131_zinterp_to_333_train': {{
    'data_dir': '{train_dir.resolve()}',
    'num_files': 1,
    'num_total_samples': {counts['train']},
    'parquet_info_path': '{info_path.resolve()}',
}},
"""
    yaml = """unified_edit:
  dataset_names:
    - cbct_131_zinterp_to_333_train
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
  weight: 1
"""
    (output_root / "bagel_131_zinterp_dataset_info_snippet.py").write_text(
        dataset_info, encoding="utf-8"
    )
    (output_root / "bagel_131_zinterp_example_config_snippet.yaml").write_text(
        yaml, encoding="utf-8"
    )


def generate_dataset(
    input_root: Path,
    output_root: Path,
    split_manifest_path: Path,
    row_group_size: int,
    shard_size: int,
    split_strategy: str,
    train_ratio: float,
    seed: int,
) -> None:
    similarity_report: list[dict[str, object]] = []
    if split_strategy == "similarity":
        bases = list_volume_bases(input_root)
        train_bases, test_bases, similarity_report = split_bases_by_similarity(
            input_root=input_root,
            bases=bases,
            train_ratio=train_ratio,
        )
        splits = {"train": train_bases, "test": test_bases}
    elif split_strategy == "random":
        bases = list_volume_bases(input_root)
        train_bases, test_bases = split_bases_random(bases, train_ratio, seed)
        splits = {"train": train_bases, "test": test_bases}
    else:
        splits = load_split_manifest(split_manifest_path)

    counts = {"train": 0, "test": 0}
    parquet_info_paths: dict[str, list[Path]] = {"train": [], "test": []}
    dataset_name = "cbct_131_zinterp_to_333"

    for split_name, volume_bases in splits.items():
        rows: list[dict[str, object]] = []
        part_idx = 0

        for base in volume_bases:
            volume_rows = build_rows_for_volume(input_root, base)
            counts[split_name] += len(volume_rows)
            rows.extend(volume_rows)

            while len(rows) >= shard_size:
                shard_rows = rows[:shard_size]
                rows = rows[shard_size:]
                parquet_path = (
                    output_root
                    / split_name
                    / dataset_name
                    / f"part-{part_idx:05d}.parquet"
                )
                write_parquet(shard_rows, parquet_path, row_group_size)
                parquet_info_paths[split_name].append(parquet_path)
                print(f"wrote {parquet_path} rows={len(shard_rows)}", flush=True)
                part_idx += 1

        if rows:
            parquet_path = (
                output_root
                / split_name
                / dataset_name
                / f"part-{part_idx:05d}.parquet"
            )
            write_parquet(rows, parquet_path, row_group_size)
            parquet_info_paths[split_name].append(parquet_path)
            print(f"wrote {parquet_path} rows={len(rows)}", flush=True)

    info_root = output_root / "parquet_info"
    write_parquet_info(
        parquet_info_paths["train"],
        info_root / "train_cbct_131_zinterp_to_333.json",
    )
    write_parquet_info(
        parquet_info_paths["test"],
        info_root / "test_cbct_131_zinterp_to_333.json",
    )
    if similarity_report:
        (output_root / "volume_similarity_report_131_zinterp.json").write_text(
            json.dumps(similarity_report, indent=2),
            encoding="utf-8",
        )
    write_split_manifest(output_root, splits, counts, split_strategy)
    write_bagel_snippets(output_root, counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--row-group-size", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument(
        "--split-strategy",
        choices=("manifest", "random", "similarity"),
        default="manifest",
        help=(
            "manifest: reuse split-manifest 131 split; random: new deterministic "
            "volume split; similarity: highest mean zinterp similarity volumes go to test."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    split_manifest = args.split_manifest.expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(input_root)
    if args.split_strategy == "manifest" and not split_manifest.exists():
        raise FileNotFoundError(split_manifest)

    generate_dataset(
        input_root=input_root,
        output_root=output_root,
        split_manifest_path=split_manifest,
        row_group_size=args.row_group_size,
        shard_size=args.shard_size,
        split_strategy=args.split_strategy,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print(f"Wrote z-interpolation CBCT data to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
