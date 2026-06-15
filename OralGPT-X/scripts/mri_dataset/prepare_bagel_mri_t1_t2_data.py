#!/usr/bin/env python3
"""Prepare BAGEL parquet shards for MRI T1 <-> T2FS modality translation."""

from __future__ import annotations

import argparse
import os
import io
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pydicom
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mri_pairs_export_slice_compare import load_by_kind  # noqa: E402
from mri_visual_plane_align import (  # noqa: E402
    align_moving_plane_to_dicom_preview,
    apply_plane_transform,
    score_plane_transforms,
)

DEFAULT_PWH_AUDIT = (
    Path(__file__).resolve().parent / "MRI_PWH_pair_audit" / "pair_audit_report.json"
)

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "dataset_T1-T2_MRI"

COHORT_PRESETS: dict[str, dict[str, object]] = {
    "Guizhou": {
        "input_root": Path(os.environ.get("ORALGPT_MRI_GUIZHOU_ROOT", "/path/to/Guizhou images")),
        "layout": "flat",
        "t1_dirname": "T1WI",
        "t2_dirname": "T2FS",
        "t1_modality": "T1WI",
        "t2_modality": "T2FS",
    },
    "Peking": {
        "input_root": Path(os.environ.get("ORALGPT_MRI_PEKING_ROOT", "/path/to/Peking")),
        "layout": "nested_series",
        "t1_dirname": "T1",
        "t2_dirname": "T2FS",
        "t1_modality": "T1",
        "t2_modality": "T2FS",
    },
    "KWC": {
        "input_root": Path(os.environ.get("ORALGPT_MRI_KWC_ROOT", "/path/to/KWC")),
        "layout": "flat",
        "t1_dirname": "T1WI",
        "t2_dirname": "T2FS",
        "t1_modality": "T1WI",
        "t2_modality": "T2FS",
    },
    "PWH": {
        "input_root": Path(os.environ.get("ORALGPT_MRI_PWH_ROOT", "/path/to/PWH")),
        "layout": "pwh_nii_dicom",
        "t1_dirname": "T1W",
        "t2_dirname": "t2fs",
        "t1_modality": "T1W",
        "t2_modality": "t2fs",
    },
}

INSTRUCTION_T1_TO_T2 = (
    "Task: MRI modality translation (T1-weighted to T2 fat-suppressed). "
    "Input: axial T1-weighted (T1WI) salivary-gland MRI slice. "
    "Target: T2 fat-suppressed (T2FS) slice at the same anatomical level. "
    "Instruction: Convert T1 contrast to T2FS appearance—emphasize fluid and "
    "soft-tissue signal expected on fat-suppressed T2, preserve lesion location, "
    "gland margins, and spatial anatomy without adding or removing pathology."
)

INSTRUCTION_T2_TO_T1 = (
    "Task: MRI modality translation (T2 fat-suppressed to T1-weighted). "
    "Input: axial T2 fat-suppressed (T2FS) salivary-gland MRI slice. "
    "Target: T1-weighted (T1WI) slice at the same anatomical level. "
    "Instruction: Convert T2FS contrast to T1 appearance—recover T1 soft-tissue "
    "and anatomical contrast, preserve lesion location, gland margins, and spatial "
    "anatomy without adding or removing pathology."
)


def slice_to_uint8_png_bytes(slice2d: np.ndarray) -> bytes:
    x = slice2d.astype(np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        image_u8 = np.zeros(slice2d.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(finite, (1.0, 99.0))
        if hi <= lo:
            lo, hi = float(finite.min()), float(finite.max())
        x = np.clip(x, lo, hi)
        image_u8 = ((x - lo) / (hi - lo + 1e-8) * 255.0).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(image_u8).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def _is_dicom_file(path: Path) -> bool:
    if not path.is_file() or path.name.startswith(".") or path.name == "DICOMDIR":
        return False
    try:
        pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def find_largest_dicom_series(subject_dir: Path) -> Path | None:
    """Pick the leaf folder with the most DICOM slices (Peking-style nesting)."""
    best_dir: Path | None = None
    best_n = 0
    for p in subject_dir.rglob("*"):
        if not p.is_file() or not _is_dicom_file(p):
            continue
        parent = p.parent
        n = sum(1 for x in parent.iterdir() if x.is_file() and _is_dicom_file(x))
        if n > best_n:
            best_n = n
            best_dir = parent
    return best_dir


@dataclass(frozen=True)
class CohortConfig:
    name: str
    input_root: Path
    layout: str  # "flat" | "nested_series"
    t1_dirname: str
    t2_dirname: str
    t1_modality: str
    t2_modality: str

    def subject_volume_dirs(self, subject_id: str) -> tuple[Path, Path]:
        if self.layout == "flat":
            return (
                self.input_root / self.t1_dirname / subject_id,
                self.input_root / self.t2_dirname / subject_id,
            )
        if self.layout == "nested_series":
            t1_subj = self.input_root / self.t1_dirname / subject_id
            t2_subj = self.input_root / self.t2_dirname / subject_id
            t1_series = find_largest_dicom_series(t1_subj)
            t2_series = find_largest_dicom_series(t2_subj)
            if t1_series is None or t2_series is None:
                raise FileNotFoundError(
                    f"No DICOM series under {subject_id} ({self.name})"
                )
            return t1_series, t2_series
        if self.layout == "pwh_nii_dicom":
            t1_nii = self.input_root / self.t1_dirname / f"{subject_id}.nii.gz"
            t2_dir = _pwh_match_t2fs_dir(self.input_root, subject_id)
            if not t1_nii.is_file() or t2_dir is None:
                raise FileNotFoundError(
                    f"Missing T1 NIfTI or t2fs for {subject_id} ({self.name})"
                )
            return t1_nii, t2_dir
        raise ValueError(f"Unknown layout: {self.layout}")

    @property
    def uses_plane_align(self) -> bool:
        return self.layout == "pwh_nii_dicom"


@dataclass(frozen=True)
class PlaneTransformQC:
    subject_id: str
    transform: str
    status: str
    num_samples: int
    dominant_fraction: float
    median_best_ncc: float
    median_margin: float
    vote_counts: dict[str, int]
    sample_records: list[dict[str, object]]

    def to_json(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "transform": self.transform,
            "status": self.status,
            "num_samples": self.num_samples,
            "dominant_fraction": self.dominant_fraction,
            "median_best_ncc": self.median_best_ncc,
            "median_margin": self.median_margin,
            "vote_counts": self.vote_counts,
            "sample_records": self.sample_records,
        }


def _pwh_match_t2fs_dir(pwh_root: Path, stem: str) -> Path | None:
    key = stem.lower()
    t2_root = pwh_root / "t2fs"
    for child in t2_root.iterdir():
        if child.is_dir() and child.name.lower() == key:
            return child
    return None


def list_subjects_from_audit(
    audit_path: Path, min_plane_ncc: float
) -> list[str]:
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    stems = []
    for case in report.get("cases", []):
        if case.get("status") != "ok":
            continue
        if float(case.get("plane_ncc", 0.0)) >= min_plane_ncc:
            stems.append(str(case["stem"]))
    return sorted(stems)


def list_subjects(config: CohortConfig) -> list[str]:
    t1_root = config.input_root / config.t1_dirname
    t2_root = config.input_root / config.t2_dirname
    if not t1_root.is_dir() or not t2_root.is_dir():
        return []
    subjects = []
    for d in sorted(t1_root.iterdir()):
        if d.is_dir() and (t2_root / d.name).is_dir():
            subjects.append(d.name)
    return subjects


def cohort_from_preset(name: str, input_root: Path | None = None) -> CohortConfig:
    preset = COHORT_PRESETS.get(name)
    if preset is None:
        raise KeyError(
            f"Unknown cohort {name!r}. Presets: {list(COHORT_PRESETS)}"
        )
    root = input_root or preset["input_root"]  # type: ignore[arg-type]
    return CohortConfig(
        name=name,
        input_root=Path(root).resolve(),
        layout=str(preset["layout"]),
        t1_dirname=str(preset["t1_dirname"]),
        t2_dirname=str(preset["t2_dirname"]),
        t1_modality=str(preset["t1_modality"]),
        t2_modality=str(preset["t2_modality"]),
    )


def split_subjects(
    subjects: list[str], train_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    shuffled = list(subjects)
    random.Random(seed).shuffle(shuffled)
    train_count = round(len(shuffled) * train_ratio)
    return sorted(shuffled[:train_count]), sorted(shuffled[train_count:])


def _median_spacing(z_mm: np.ndarray) -> float:
    z = np.asarray(z_mm, dtype=np.float64)
    if z.size < 2:
        return 5.0
    dz = np.abs(np.diff(np.sort(z)))
    dz = dz[np.isfinite(dz) & (dz > 1e-6)]
    if dz.size == 0:
        return 5.0
    return float(np.median(dz))


def pair_slice_indices_by_z(
    z_src: np.ndarray, z_tgt: np.ndarray, max_delta_mm: float | None = None
) -> list[tuple[int, int, float, float, float]]:
    """Map each source slice to closest target slice within z tolerance."""
    z_src = np.asarray(z_src, dtype=np.float64)
    z_tgt = np.asarray(z_tgt, dtype=np.float64)
    if max_delta_mm is None:
        max_delta_mm = max(
            1.5,
            0.6 * min(_median_spacing(z_src), _median_spacing(z_tgt)),
        )

    pairs: list[tuple[int, int, float, float, float]] = []
    for i in range(int(z_src.shape[0])):
        j = int(np.argmin(np.abs(z_tgt - z_src[i])))
        dz = float(abs(z_tgt[j] - z_src[i]))
        if dz <= max_delta_mm:
            pairs.append((i, j, float(z_src[i]), float(z_tgt[j]), dz))
    return pairs


def _load_volumes(config: CohortConfig, subject_id: str):
    t1_path, t2_path = config.subject_volume_dirs(subject_id)
    if config.layout == "pwh_nii_dicom":
        return load_by_kind(t1_path, "nii"), load_by_kind(t2_path, "dicom"), t1_path, t2_path
    return load_by_kind(t1_path, "dicom"), load_by_kind(t2_path, "dicom"), t1_path, t2_path


def _sample_pair_indices(
    pairs: list[tuple[int, int, float, float, float]], max_samples: int
) -> list[tuple[int, int, float, float, float]]:
    if len(pairs) <= max_samples:
        return pairs
    sample_positions = np.linspace(0, len(pairs) - 1, num=max_samples)
    return [pairs[int(round(pos))] for pos in sample_positions]


def estimate_pwh_subject_plane_transform(
    subject_id: str,
    vol_t1,
    vol_t2,
    pairs: list[tuple[int, int, float, float, float]],
    *,
    max_samples: int,
    min_vote_fraction: float,
    min_median_ncc: float,
    min_median_margin: float,
) -> PlaneTransformQC:
    """Estimate one fixed T1->t2fs plane transform for a PWH subject."""
    sampled_pairs = _sample_pair_indices(pairs, max_samples=max_samples)
    vote_counts: Counter[str] = Counter()
    best_scores: list[float] = []
    margins: list[float] = []
    sample_records: list[dict[str, object]] = []

    for t1_idx, t2_idx, z_t1, z_t2, dz in sampled_pairs:
        moving = vol_t1.data[t1_idx].astype(np.float32)
        reference = vol_t2.data[t2_idx].astype(np.float32)
        scores = score_plane_transforms(moving, reference)
        if not scores:
            continue
        best_name, best_score = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else float("-inf")
        margin = float(best_score - second_score)
        vote_counts[best_name] += 1
        best_scores.append(float(best_score))
        margins.append(margin)
        sample_records.append(
            {
                "t1_slice_index": int(t1_idx),
                "t2_slice_index": int(t2_idx),
                "t1_z_mm": float(z_t1),
                "t2_z_mm": float(z_t2),
                "z_delta_mm": float(dz),
                "best_transform": best_name,
                "best_ncc": float(best_score),
                "second_best_ncc": float(second_score),
                "best_margin": margin,
            }
        )

    if not vote_counts:
        return PlaneTransformQC(
            subject_id=subject_id,
            transform="identity",
            status="reject:no_transform_scores",
            num_samples=0,
            dominant_fraction=0.0,
            median_best_ncc=float("-inf"),
            median_margin=float("-inf"),
            vote_counts={},
            sample_records=[],
        )

    transform, transform_votes = vote_counts.most_common(1)[0]
    dominant_fraction = float(transform_votes / sum(vote_counts.values()))
    median_best_ncc = float(np.median(np.asarray(best_scores, dtype=np.float64)))
    median_margin = float(np.median(np.asarray(margins, dtype=np.float64)))
    reject_reasons: list[str] = []
    if dominant_fraction < min_vote_fraction:
        reject_reasons.append("low_vote_fraction")
    if median_best_ncc < min_median_ncc:
        reject_reasons.append("low_median_ncc")
    if min_median_margin > 0.0 and median_margin < min_median_margin:
        reject_reasons.append("ambiguous_transform_margin")
    status = "include" if not reject_reasons else "reject:" + ",".join(reject_reasons)

    return PlaneTransformQC(
        subject_id=subject_id,
        transform=transform,
        status=status,
        num_samples=len(sample_records),
        dominant_fraction=round(dominant_fraction, 4),
        median_best_ncc=round(median_best_ncc, 4),
        median_margin=round(median_margin, 4),
        vote_counts=dict(vote_counts.most_common()),
        sample_records=sample_records,
    )


def _png_pair_for_slices(
    config: CohortConfig,
    direction: str,
    src_slice: np.ndarray,
    tgt_slice: np.ndarray,
    plane_transform: str | None = None,
) -> list[bytes]:
    if not config.uses_plane_align:
        return [
            slice_to_uint8_png_bytes(src_slice),
            slice_to_uint8_png_bytes(tgt_slice),
        ]
    if direction == "t1_to_t2":
        if plane_transform is None:
            t1_al, _, _ = align_moving_plane_to_dicom_preview(src_slice, tgt_slice)
        else:
            t1_al = apply_plane_transform(src_slice, plane_transform)
        return [slice_to_uint8_png_bytes(t1_al), slice_to_uint8_png_bytes(tgt_slice)]
    if plane_transform is None:
        t1_al, _, _ = align_moving_plane_to_dicom_preview(tgt_slice, src_slice)
    else:
        t1_al = apply_plane_transform(tgt_slice, plane_transform)
    return [slice_to_uint8_png_bytes(src_slice), slice_to_uint8_png_bytes(t1_al)]


def build_rows_for_subject(
    config: CohortConfig,
    subject_id: str,
    direction: str,
    plane_qc_by_subject: dict[str, PlaneTransformQC] | None = None,
    *,
    pwh_transform_samples: int = 9,
    pwh_min_transform_vote_fraction: float = 0.75,
    pwh_min_transform_ncc: float = 0.55,
    pwh_min_transform_margin: float = 0.0,
    pwh_filter_unstable_plane_transform: bool = False,
) -> list[dict[str, object]]:
    vol_t1, vol_t2, t1_path, t2_path = _load_volumes(config, subject_id)
    plane_qc: PlaneTransformQC | None = None
    if config.uses_plane_align:
        if plane_qc_by_subject is not None and subject_id in plane_qc_by_subject:
            plane_qc = plane_qc_by_subject[subject_id]
        else:
            plane_pairs = pair_slice_indices_by_z(vol_t1.z_mm, vol_t2.z_mm)
            plane_qc = estimate_pwh_subject_plane_transform(
                subject_id,
                vol_t1,
                vol_t2,
                plane_pairs,
                max_samples=pwh_transform_samples,
                min_vote_fraction=pwh_min_transform_vote_fraction,
                min_median_ncc=pwh_min_transform_ncc,
                min_median_margin=pwh_min_transform_margin,
            )
            if plane_qc_by_subject is not None:
                plane_qc_by_subject[subject_id] = plane_qc

        if pwh_filter_unstable_plane_transform and plane_qc.status != "include":
            return []

    if direction == "t1_to_t2":
        src_vol, tgt_vol = vol_t1, vol_t2
        instruction = INSTRUCTION_T1_TO_T2
        src_mod, tgt_mod = config.t1_modality, config.t2_modality
    elif direction == "t2_to_t1":
        src_vol, tgt_vol = vol_t2, vol_t1
        instruction = INSTRUCTION_T2_TO_T1
        src_mod, tgt_mod = config.t2_modality, config.t1_modality
    else:
        raise ValueError(direction)

    pairs = pair_slice_indices_by_z(src_vol.z_mm, tgt_vol.z_mm)
    rows: list[dict[str, object]] = []
    slug = _cohort_slug(config.name)

    for src_idx, tgt_idx, z_src, z_tgt, dz in pairs:
        src_slice = src_vol.data[src_idx].astype(np.float32)
        tgt_slice = tgt_vol.data[tgt_idx].astype(np.float32)
        row = {
            "image_list": _png_pair_for_slices(
                config,
                direction,
                src_slice,
                tgt_slice,
                plane_transform=plane_qc.transform if plane_qc is not None else None,
            ),
            "instruction_list": [[instruction]],
            "cohort": config.name,
            "source_modality": src_mod,
            "target_modality": tgt_mod,
            "direction": direction,
            "volume_id": subject_id,
            "source_slice_index": int(src_idx),
            "target_slice_index": int(tgt_idx),
            "source_z_mm": z_src,
            "target_z_mm": z_tgt,
            "z_delta_mm": dz,
            "t1_series_path": str(t1_path),
            "t2_series_path": str(t2_path),
            "pair_id": (
                f"{slug}_{subject_id}_{src_mod}_{src_idx:03d}_to_"
                f"{tgt_mod}_{tgt_idx:03d}"
            ),
        }
        if config.uses_plane_align:
            row["plane_align"] = "nifti_to_dicom_heuristic"
            row["plane_transform_scope"] = "subject"
            row["plane_transform"] = plane_qc.transform if plane_qc is not None else None
            row["plane_transform_status"] = plane_qc.status if plane_qc is not None else None
            row["plane_transform_dominant_fraction"] = (
                plane_qc.dominant_fraction if plane_qc is not None else None
            )
            row["plane_transform_median_ncc"] = (
                plane_qc.median_best_ncc if plane_qc is not None else None
            )
            row["plane_transform_median_margin"] = (
                plane_qc.median_margin if plane_qc is not None else None
            )
        rows.append(row)
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


def write_pwh_plane_transform_qc(
    plane_qc_by_subject: dict[str, PlaneTransformQC], output_path: Path
) -> None:
    status_counts = Counter(qc.status for qc in plane_qc_by_subject.values())
    transform_counts = Counter(qc.transform for qc in plane_qc_by_subject.values())
    report = {
        "note": (
            "PWH T1W NIfTI is aligned to t2fs DICOM with one fixed in-plane "
            "transform per subject. This avoids per-slice flip/rotation changes."
        ),
        "status_counts": dict(status_counts.most_common()),
        "transform_counts": dict(transform_counts.most_common()),
        "subjects": [
            qc.to_json()
            for qc in sorted(plane_qc_by_subject.values(), key=lambda item: item.subject_id)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_split_manifest(
    output_root: Path,
    config: CohortConfig,
    split_map: dict[str, list[str]],
    counts: dict[str, dict[str, int]],
    train_ratio: float,
    qc_notes: dict[str, object] | None = None,
) -> None:
    manifest = {
        "split_policy": "volume-level deterministic random split",
        "train_ratio": train_ratio,
        "cohort": config.name,
        "data_root": str(config.input_root),
        "layout": config.layout,
        "t1_dirname": config.t1_dirname,
        "t2_dirname": config.t2_dirname,
        "qc": qc_notes or {},
        "pairing_rules": {
            "t1_to_t2": (
                f"{config.t1_modality} slice i -> closest {config.t2_modality} slice j "
                "by physical z (|Δz| <= tolerance)"
            ),
            "t2_to_t1": (
                f"{config.t2_modality} slice i -> closest {config.t1_modality} slice j "
                "by physical z (|Δz| <= tolerance)"
            ),
        },
        "datasets": split_map,
        "counts": counts,
    }
    (output_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def write_bagel_snippets(
    output_root: Path, cohort_slug: str, counts: dict[str, dict[str, int]]
) -> None:
    lines = [f"# {cohort_slug} cohort — add under DATASET_INFO['unified_edit']."]
    dataset_names: list[str] = []
    for direction, key in [
        ("t1_to_t2", "mri_t1_to_t2"),
        ("t2_to_t1", "mri_t2_to_t1"),
    ]:
        train_dir = output_root / "train" / key
        info_path = output_root / "parquet_info" / f"train_{direction}.json"
        num_files = len(list(train_dir.glob("part-*.parquet"))) if train_dir.is_dir() else 0
        ds_name = f"{cohort_slug}_{key}_train"
        dataset_names.append(ds_name)
        lines.append(
            f"'{ds_name}': {{\n"
            f"    'data_dir': '{train_dir.resolve()}',\n"
            f"    'num_files': {num_files},\n"
            f"    'num_total_samples': {counts['train'][direction]},\n"
            f"    'parquet_info_path': '{info_path.resolve()}',\n"
            f"}},"
        )
    (output_root / "bagel_dataset_info_snippet.py").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    names_yaml = "\n".join(f"    - {n}" for n in dataset_names)
    yaml = f"""# {cohort_slug} cohort
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
    (output_root / "bagel_example_config_snippet.yaml").write_text(yaml, encoding="utf-8")


def _cohort_slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def update_root_cohort_index(
    dataset_root: Path, config: CohortConfig, counts: dict[str, dict[str, int]]
) -> None:
    index_path = dataset_root / "cohorts.json"
    index: dict = {"cohorts": {}}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    index.setdefault("cohorts", {})[config.name] = {
        "path": config.name,
        "data_source": str(config.input_root),
        "layout": config.layout,
        "modalities": [config.t1_modality, config.t2_modality],
        "counts": counts,
        "split_manifest": f"{config.name}/split_manifest.json",
    }
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def generate_dataset(
    config: CohortConfig,
    output_root: Path,
    train_ratio: float,
    seed: int,
    row_group_size: int,
    shard_size: int,
    subjects: list[str] | None = None,
    qc_notes: dict[str, object] | None = None,
    pwh_transform_samples: int = 9,
    pwh_min_transform_vote_fraction: float = 0.75,
    pwh_min_transform_ncc: float = 0.55,
    pwh_min_transform_margin: float = 0.0,
    pwh_filter_unstable_plane_transform: bool = False,
) -> None:
    if subjects is None:
        subjects = list_subjects(config)
    train_subjects, test_subjects = split_subjects(subjects, train_ratio, seed)
    plane_qc_by_subject: dict[str, PlaneTransformQC] = {}

    split_map = {"train": train_subjects, "test": test_subjects}
    counts: dict[str, dict[str, int]] = {
        "train": {"t1_to_t2": 0, "t2_to_t1": 0},
        "test": {"t1_to_t2": 0, "t2_to_t1": 0},
    }
    parquet_info_paths: dict[str, list[Path]] = {
        "train_t1_to_t2": [],
        "train_t2_to_t1": [],
        "test_t1_to_t2": [],
        "test_t2_to_t1": [],
    }

    dataset_keys = {
        "t1_to_t2": "mri_t1_to_t2",
        "t2_to_t1": "mri_t2_to_t1",
    }

    for split_name, subject_list in [("train", train_subjects), ("test", test_subjects)]:
        for direction, dataset_name in dataset_keys.items():
            rows: list[dict[str, object]] = []
            part_idx = 0

            for subject_id in subject_list:
                try:
                    volume_rows = build_rows_for_subject(
                        config,
                        subject_id,
                        direction,
                        plane_qc_by_subject=plane_qc_by_subject
                        if config.uses_plane_align
                        else None,
                        pwh_transform_samples=pwh_transform_samples,
                        pwh_min_transform_vote_fraction=pwh_min_transform_vote_fraction,
                        pwh_min_transform_ncc=pwh_min_transform_ncc,
                        pwh_min_transform_margin=pwh_min_transform_margin,
                        pwh_filter_unstable_plane_transform=(
                            pwh_filter_unstable_plane_transform
                        ),
                    )
                except Exception as exc:
                    print(
                        f"skip {split_name}/{subject_id}/{direction}: {exc}",
                        flush=True,
                    )
                    continue
                counts[split_name][direction] += len(volume_rows)
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
                    parquet_info_paths[f"{split_name}_{direction}"].append(parquet_path)
                    print(
                        f"wrote {parquet_path} rows={len(shard_rows)}",
                        flush=True,
                    )
                    part_idx += 1

            if rows:
                parquet_path = (
                    output_root
                    / split_name
                    / dataset_name
                    / f"part-{part_idx:05d}.parquet"
                )
                write_parquet(rows, parquet_path, row_group_size)
                parquet_info_paths[f"{split_name}_{direction}"].append(parquet_path)
                print(f"wrote {parquet_path} rows={len(rows)}", flush=True)

    info_root = output_root / "parquet_info"
    for key, paths in parquet_info_paths.items():
        write_parquet_info(paths, info_root / f"{key}.json")

    if config.uses_plane_align:
        write_pwh_plane_transform_qc(
            plane_qc_by_subject, output_root / "pwh_plane_transform_qc.json"
        )

    write_split_manifest(
        output_root, config, split_map, counts, train_ratio, qc_notes=qc_notes
    )
    write_bagel_snippets(output_root, _cohort_slug(config.name), counts)
    update_root_cohort_index(output_root.parent, config, counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Override raw data root (default: preset for --cohort)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Cohort output dir (default: dataset_T1-T2_MRI/<cohort>)",
    )
    parser.add_argument(
        "--cohort",
        type=str,
        default="Guizhou",
        choices=list(COHORT_PRESETS),
        help=f"Preset cohort ({', '.join(COHORT_PRESETS)})",
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=None,
        help="Optional audit JSON to filter subjects (e.g. PWH plane_ncc)",
    )
    parser.add_argument(
        "--min-plane-ncc",
        type=float,
        default=None,
        help="With --audit-report: keep subjects with plane_ncc >= this value",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--row-group-size", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument(
        "--pwh-transform-samples",
        type=int,
        default=9,
        help="PWH only: number of z-paired slices used to vote for subject-level plane transform",
    )
    parser.add_argument(
        "--pwh-min-transform-vote-fraction",
        type=float,
        default=0.75,
        help="PWH only: QC threshold for dominant subject-level plane transform vote",
    )
    parser.add_argument(
        "--pwh-min-transform-ncc",
        type=float,
        default=0.55,
        help="PWH only: QC threshold for median transform NCC across sampled slices",
    )
    parser.add_argument(
        "--pwh-min-transform-margin",
        type=float,
        default=0.0,
        help=(
            "PWH only: optional QC threshold for median best-vs-second transform "
            "NCC margin. Default 0 disables margin rejection because square-plane "
            "transform candidates can tie."
        ),
    )
    parser.add_argument(
        "--pwh-filter-unstable-plane-transform",
        action="store_true",
        help="PWH only: drop subjects whose fixed plane transform QC fails",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cohort_name = args.cohort.strip()
    config = cohort_from_preset(
        cohort_name,
        args.input_root.expanduser().resolve() if args.input_root else None,
    )
    if args.output_root is None:
        output_root = DEFAULT_OUTPUT_ROOT / cohort_name
    else:
        output_root = args.output_root.expanduser().resolve()
    if not config.input_root.exists():
        raise FileNotFoundError(config.input_root)

    subjects: list[str] | None = None
    qc_notes: dict[str, object] | None = None
    audit_path = (
        args.audit_report.expanduser().resolve()
        if args.audit_report is not None
        else None
    )
    min_ncc = args.min_plane_ncc
    if cohort_name == "PWH" and audit_path is None:
        audit_path = DEFAULT_PWH_AUDIT.resolve()
        min_ncc = 0.55 if min_ncc is None else min_ncc
    if audit_path is not None:
        if min_ncc is None:
            raise ValueError("--audit-report (or PWH default) requires --min-plane-ncc")
        subjects = list_subjects_from_audit(audit_path, min_ncc)
        qc_notes = {
            "audit_report": str(audit_path),
            "min_plane_ncc": min_ncc,
            "subject_count": len(subjects),
        }
        if cohort_name == "PWH":
            qc_notes["plane_transform_policy"] = (
                "fixed subject-level T1W NIfTI -> t2fs DICOM transform"
            )
            qc_notes["plane_transform_qc_path"] = str(
                output_root / "pwh_plane_transform_qc.json"
            )
            qc_notes["pwh_transform_samples"] = args.pwh_transform_samples
            qc_notes["pwh_min_transform_vote_fraction"] = (
                args.pwh_min_transform_vote_fraction
            )
            qc_notes["pwh_min_transform_ncc"] = args.pwh_min_transform_ncc
            qc_notes["pwh_min_transform_margin"] = args.pwh_min_transform_margin
            qc_notes["pwh_filter_unstable_plane_transform"] = (
                args.pwh_filter_unstable_plane_transform
            )

    generate_dataset(
        config=config,
        output_root=output_root,
        train_ratio=args.train_ratio,
        seed=args.seed,
        row_group_size=args.row_group_size,
        shard_size=args.shard_size,
        subjects=subjects,
        qc_notes=qc_notes,
        pwh_transform_samples=args.pwh_transform_samples,
        pwh_min_transform_vote_fraction=args.pwh_min_transform_vote_fraction,
        pwh_min_transform_ncc=args.pwh_min_transform_ncc,
        pwh_min_transform_margin=args.pwh_min_transform_margin,
        pwh_filter_unstable_plane_transform=args.pwh_filter_unstable_plane_transform,
    )
    print(f"Wrote BAGEL MRI T1/T2 data [{cohort_name}] to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
