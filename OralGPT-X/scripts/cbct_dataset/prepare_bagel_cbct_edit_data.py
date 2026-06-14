#!/usr/bin/env python3
"""Prepare BAGEL image-editing parquet shards for CBCT dose enhancement."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import random
import re
import shutil
from functools import lru_cache
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
DOSES = ("78", "123", "131")
Z_MATCH_TOLERANCE_MM = 0.0001
PREPROCESS_SLICE_START = 50
PREPROCESS_SLICE_STOP = 240

INSTRUCTION_78 = (
    "Task: Standard-dose CBCT image enhancement through noise suppression. "
    "Input: Low-dose dental CBCT slice acquired at 78 mGy.cm^2 using reduced "
    "mAs and a voxel size of 0.2 mm. "
    "Target: Standard-dose-equivalent CBCT image corresponding to 333 mGy.cm^2 "
    "with the same voxel size of 0.2 mm. "
    "Instruction: Enhance the low-dose CBCT image by suppressing noise caused "
    "by reduced mAs while improving contrast, edge definition, and visibility "
    "of the boundaries of hard tissue and soft tissue space, including teeth, "
    "alveolar bony structures, periodontal ligament spaces, and mandibular "
    "canals. Preserve the exact patient-specific anatomy, pathology, and "
    "tooth-canal spatial relationships. Do not introduce, remove, or alter "
    "anatomical structures or pathology. The enhanced image should approximate "
    "the appearance and diagnostic quality of a standard-dose CBCT image while "
    "avoiding hallucinated details."
)

INSTRUCTION_131 = (
    "Task: Standard-dose CBCT image enhancement through noise suppression and "
    "spatial-resolution improvement. "
    "Input: Low-dose dental CBCT slice acquired at 131 mGy.cm^2 using reduced "
    "mAs with a voxel size of 0.4 mm. "
    "Target: Standard-dose-equivalent CBCT image corresponding to 333 mGy.cm^2 "
    "with a smaller voxel size of 0.2 mm. "
    "Instruction: Enhance the low-dose CBCT image by suppressing noise caused "
    "by reduced mAs and improving spatial resolution while improving contrast, "
    "edge definition, and visibility of the boundaries of hard tissue and soft "
    "tissue space, including teeth, alveolar bony structures, periodontal "
    "ligament spaces, and mandibular canals. Preserve the exact patient-specific "
    "anatomy, pathology, and tooth-canal spatial relationships. Do not "
    "introduce, remove, or alter anatomical structures or pathology. The "
    "enhanced image should approximate the appearance and diagnostic quality of "
    "a standard-dose CBCT image while avoiding hallucinated details."
)

INSTRUCTION_123 = (
    "Task: Standard-dose CBCT image enhancement through noise suppression. "
    "Input: Low-dose dental CBCT slice acquired at 123 mGy.cm^2 using reduced "
    "mAs with a voxel size of 0.15 mm. "
    "Target: Standard-dose-equivalent CBCT image corresponding to 333 mGy.cm^2 "
    "with a larger voxel size of 0.2 mm. "
    "Instruction: Enhance the low-dose CBCT image by suppressing noise caused "
    "by reduced mAs while improving contrast, edge definition, and visibility "
    "of the boundaries of hard tissue and soft tissue space, including teeth, "
    "alveolar bony structures, periodontal ligament spaces, and mandibular "
    "canals. Preserve the exact patient-specific anatomy, pathology, and "
    "tooth-canal spatial relationships. Do not introduce, remove, or alter "
    "anatomical structures or pathology. The enhanced image should approximate "
    "the appearance and diagnostic quality of a standard-dose CBCT image while "
    "avoiding hallucinated details."
)

INSTRUCTIONS = {
    "78": INSTRUCTION_78,
    "123": INSTRUCTION_123,
    "131": INSTRUCTION_131,
}


def dicom_to_uint8_png_bytes(
    dcm_path: Path,
    output_size: tuple[int, int] | None = None,
) -> bytes:
    """Convert a single-frame DICOM slice to grayscale PNG bytes for BAGEL/PIL."""
    ds = pydicom.dcmread(str(dcm_path), force=True)
    image_u8 = dicom_to_preprocessed_uint8(ds)

    buf = io.BytesIO()
    image = Image.fromarray(image_u8)
    if output_size is not None and image.size != output_size:
        image = image.resize(output_size, Image.Resampling.LANCZOS)
    image.save(buf, format="PNG", compress_level=1)
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


@lru_cache(maxsize=None)
def image_position_z(dcm_path: Path) -> float | None:
    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is None or len(ipp) < 3:
        return None
    return float(ipp[2])


@lru_cache(maxsize=None)
def dicom_image_size(dcm_path: Path) -> tuple[int, int]:
    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
    return int(ds.Columns), int(ds.Rows)


def is_kept_after_preprocess(slice_idx: int) -> bool:
    return PREPROCESS_SLICE_START <= slice_idx <= PREPROCESS_SLICE_STOP


def preprocess_slice_index(original_slice_idx: int) -> int:
    if not is_kept_after_preprocess(original_slice_idx):
        raise ValueError(f"Slice {original_slice_idx} is outside the preprocessed range")
    return original_slice_idx - PREPROCESS_SLICE_START + 1


def target_index_for_123(low_idx: int, low_path: Path, target_slices: dict[int, Path]) -> int | None:
    low_z = image_position_z(low_path)
    if low_z is None:
        return None

    for expected_idx in (2 * low_idx, low_idx * 3 // 4 if low_idx % 4 == 0 else None):
        if expected_idx is None:
            continue
        target_path = target_slices.get(expected_idx)
        if target_path is not None:
            target_z = image_position_z(target_path)
            if (
                target_z is not None
                and abs(low_z - target_z) <= Z_MATCH_TOLERANCE_MM
            ):
                return expected_idx

    return None


def target_index_for_dose(
    dose: str,
    low_idx: int,
    low_path: Path,
    target_slices: dict[int, Path],
) -> int | None:
    if dose == "78":
        return low_idx
    if dose == "131":
        return 2 * low_idx - 1
    if dose == "123":
        return target_index_for_123(low_idx, low_path, target_slices)
    raise ValueError(f"Unsupported dose: {dose}")


def list_volume_bases(input_root: Path, dose: str) -> list[str]:
    bases = []
    folder_names = {path.name for path in input_root.iterdir() if path.is_dir()}
    for name in sorted(folder_names):
        if not name.endswith(f"_{dose}"):
            continue
        base = name.rsplit("_", 1)[0]
        if f"{base}_333" in folder_names:
            bases.append(base)
    return bases


def split_bases(bases: list[str], train_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    shuffled = list(bases)
    random.Random(seed).shuffle(shuffled)
    train_count = round(len(shuffled) * train_ratio)
    train = sorted(shuffled[:train_count])
    test = sorted(shuffled[train_count:])
    return train, test


def pair_similarity(source_png: bytes, target_png: bytes) -> float:
    source = Image.open(io.BytesIO(source_png)).convert("L")
    target = Image.open(io.BytesIO(target_png)).convert("L")
    if source.size != target.size:
        source = source.resize(target.size, Image.Resampling.LANCZOS)

    source_arr = np.asarray(source, dtype=np.float32).ravel()
    target_arr = np.asarray(target, dtype=np.float32).ravel()
    source_arr -= float(source_arr.mean())
    target_arr -= float(target_arr.mean())

    denominator = float(np.linalg.norm(source_arr) * np.linalg.norm(target_arr))
    if denominator == 0:
        return 0.0
    return float(np.dot(source_arr, target_arr) / denominator)


def volume_similarity(
    input_root: Path,
    base: str,
    dose: str,
) -> tuple[float, int]:
    rows = build_rows_for_volume(input_root, base, dose)
    if not rows:
        return float("-inf"), 0
    scores = [
        pair_similarity(row["image_list"][0], row["image_list"][1])
        for row in rows
    ]
    return float(sum(scores) / len(scores)), len(rows)


def split_bases_by_similarity(
    input_root: Path,
    bases: list[str],
    dose: str,
    train_ratio: float,
) -> tuple[list[str], list[str], list[dict[str, object]]]:
    test_count = len(bases) - round(len(bases) * train_ratio)
    if test_count <= 0:
        return sorted(bases), [], []

    report_rows = []
    for idx, base in enumerate(bases, start=1):
        avg_similarity, num_pairs = volume_similarity(input_root, base, dose)
        report_rows.append(
            {
                "dose": dose,
                "volume_id": base,
                "avg_similarity": avg_similarity,
                "num_pairs": num_pairs,
            }
        )
        print(
            f"similarity {dose} {idx}/{len(bases)} {base}: "
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


def build_rows_for_volume(input_root: Path, base: str, dose: str) -> list[dict[str, object]]:
    low_dir = input_root / f"{base}_{dose}"
    target_dir = input_root / f"{base}_333"
    low_slices = indexed_dicoms(low_dir)
    target_slices = indexed_dicoms(target_dir)
    instruction = INSTRUCTIONS[dose]

    rows = []
    for low_idx in sorted(low_slices):
        if not is_kept_after_preprocess(low_idx):
            continue
        target_idx = target_index_for_dose(dose, low_idx, low_slices[low_idx], target_slices)
        if target_idx is None:
            continue
        if not is_kept_after_preprocess(target_idx):
            continue
        target_path = target_slices.get(target_idx)
        if target_path is None:
            continue
        source_output_size = dicom_image_size(target_path) if dose == "123" else None
        source_preprocessed_idx = preprocess_slice_index(low_idx)
        target_preprocessed_idx = preprocess_slice_index(target_idx)

        rows.append(
            {
                "image_list": [
                    dicom_to_uint8_png_bytes(
                        low_slices[low_idx],
                        output_size=source_output_size,
                    ),
                    dicom_to_uint8_png_bytes(target_path),
                ],
                "instruction_list": [[instruction]],
                "source_dose": dose,
                "target_dose": "333",
                "volume_id": base,
                "source_slice_index": source_preprocessed_idx,
                "target_slice_index": target_preprocessed_idx,
                "source_original_slice_index": low_idx,
                "target_original_slice_index": target_idx,
                "pair_id": (
                    f"{base}_{dose}_{source_preprocessed_idx:03d}"
                    f"_to_333_{target_preprocessed_idx:03d}"
                    f"_orig_{low_idx:03d}_to_{target_idx:03d}"
                ),
            }
        )
    return rows


def write_parquet(rows: list[dict[str, object]], output_path: Path, row_group_size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path, row_group_size=row_group_size)


def write_parquet_info(parquet_paths: list[Path], output_path: Path) -> None:
    info = {}
    for path in parquet_paths:
        info[str(path.resolve())] = {"num_row_groups": pq.ParquetFile(path).num_row_groups}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def clean_previous_outputs(output_root: Path, doses: tuple[str, ...]) -> None:
    for dose in doses:
        for split_name in ("train", "test"):
            dataset_dir = output_root / split_name / f"cbct_{dose}_to_333"
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)

            info_path = output_root / "parquet_info" / f"{split_name}_cbct_{dose}_to_333.json"
            if info_path.exists():
                info_path.unlink()


def write_split_manifest(
    output_root: Path,
    split_map: dict[str, dict[str, list[str]]],
    counts: dict[str, dict[str, int]],
    train_ratio: float,
    doses: tuple[str, ...],
    split_strategy: str,
) -> None:
    pairing_rules = {
        "78": "source slice k -> target slice k",
        "123": (
            "exact ImagePositionPatient[2] match; common cases map source "
            "slice 4*n -> target slice 3*n"
        ),
        "131": "source slice k -> target slice 2*k-1",
    }
    manifest = {
        "split_policy": f"volume-level {split_strategy} split",
        "train_ratio": train_ratio,
        "pairing_rules": {f"{dose}_to_333": pairing_rules[dose] for dose in doses},
        "datasets": split_map,
        "counts": counts,
    }
    (output_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def write_bagel_snippets(
    output_root: Path,
    counts: dict[str, dict[str, int]],
    doses: tuple[str, ...],
) -> None:
    dataset_info_lines = ["# Add these entries under DATASET_INFO['unified_edit']."]
    for dose in doses:
        train_dir = output_root / "train" / f"cbct_{dose}_to_333"
        info_path = output_root / "parquet_info" / f"train_cbct_{dose}_to_333.json"
        dataset_info_lines.extend(
            [
                f"'cbct_{dose}_to_333_train': {{",
                f"    'data_dir': '{train_dir.resolve()}',",
                "    'num_files': 1,",
                f"    'num_total_samples': {counts['train'][dose]},",
                f"    'parquet_info_path': '{info_path.resolve()}',",
                "},",
            ]
        )
    dataset_info = "\n".join(dataset_info_lines) + "\n"

    dataset_names = "\n".join(f"    - cbct_{dose}_to_333_train" for dose in doses)
    num_used_data = "\n".join("    - 1" for _ in doses)
    yaml = f"""unified_edit:
  dataset_names:
{dataset_names}
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
{num_used_data}
  weight: 1
"""

    (output_root / "bagel_dataset_info_snippet.py").write_text(dataset_info, encoding="utf-8")
    (output_root / "bagel_example_config_snippet.yaml").write_text(yaml, encoding="utf-8")


def generate_dataset(
    input_root: Path,
    output_root: Path,
    doses: tuple[str, ...],
    train_ratio: float,
    seed: int,
    row_group_size: int,
    shard_size: int,
    split_strategy: str,
) -> None:
    clean_previous_outputs(output_root, doses)

    split_map: dict[str, dict[str, list[str]]] = {}
    similarity_reports: dict[str, list[dict[str, object]]] = {}
    counts: dict[str, dict[str, int]] = {
        "train": {dose: 0 for dose in doses},
        "test": {dose: 0 for dose in doses},
    }
    parquet_info_paths: dict[str, list[Path]] = {
        f"{split_name}_{dose}": []
        for split_name in ("train", "test")
        for dose in doses
    }

    for dose in doses:
        bases = list_volume_bases(input_root, dose)
        if split_strategy == "similarity":
            train_bases, test_bases, report_rows = split_bases_by_similarity(
                input_root=input_root,
                bases=bases,
                dose=dose,
                train_ratio=train_ratio,
            )
            similarity_reports[dose] = report_rows
        else:
            train_bases, test_bases = split_bases(bases, train_ratio, seed + int(dose))
        split_map[dose] = {"train": train_bases, "test": test_bases}

        for split_name, split_bases_list in [("train", train_bases), ("test", test_bases)]:
            rows = []
            part_idx = 0
            for base in split_bases_list:
                volume_rows = build_rows_for_volume(input_root, base, dose)
                counts[split_name][dose] += len(volume_rows)
                rows.extend(volume_rows)

                while len(rows) >= shard_size:
                    shard_rows = rows[:shard_size]
                    rows = rows[shard_size:]
                    dataset_name = f"cbct_{dose}_to_333"
                    parquet_path = (
                        output_root
                        / split_name
                        / dataset_name
                        / f"part-{part_idx:05d}.parquet"
                    )
                    write_parquet(shard_rows, parquet_path, row_group_size)
                    parquet_info_paths[f"{split_name}_{dose}"].append(parquet_path)
                    print(
                        f"wrote {parquet_path} rows={len(shard_rows)}",
                        flush=True,
                    )
                    part_idx += 1

            if rows:
                dataset_name = f"cbct_{dose}_to_333"
                parquet_path = (
                    output_root
                    / split_name
                    / dataset_name
                    / f"part-{part_idx:05d}.parquet"
                )
                write_parquet(rows, parquet_path, row_group_size)
                parquet_info_paths[f"{split_name}_{dose}"].append(parquet_path)
                print(f"wrote {parquet_path} rows={len(rows)}", flush=True)

    info_root = output_root / "parquet_info"
    for split_name in ("train", "test"):
        for dose in doses:
            write_parquet_info(
                parquet_info_paths[f"{split_name}_{dose}"],
                info_root / f"{split_name}_cbct_{dose}_to_333.json",
            )

    if similarity_reports:
        similarity_path = output_root / "volume_similarity_report.json"
        similarity_path.write_text(
            json.dumps(similarity_reports, indent=2),
            encoding="utf-8",
        )

    write_split_manifest(output_root, split_map, counts, train_ratio, doses, split_strategy)
    write_bagel_snippets(output_root, counts, doses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--row-group-size", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument(
        "--split-strategy",
        choices=("random", "similarity"),
        default="random",
        help=(
            "random: deterministic volume split; similarity: put volumes with "
            "highest mean paired-slice similarity into test."
        ),
    )
    parser.add_argument(
        "--doses",
        nargs="+",
        choices=DOSES,
        default=list(DOSES),
        help="Dose groups to build. Default: 78 123 131.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(input_root)

    generate_dataset(
        input_root=input_root,
        output_root=output_root,
        doses=tuple(args.doses),
        train_ratio=args.train_ratio,
        seed=args.seed,
        row_group_size=args.row_group_size,
        shard_size=args.shard_size,
        split_strategy=args.split_strategy,
    )
    print(f"Wrote BAGEL CBCT data to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
