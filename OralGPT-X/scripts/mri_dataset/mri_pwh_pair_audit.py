#!/usr/bin/env python3
"""Audit all PWH NIfTI(T1W) vs DICOM(t2fs) pairs: preview PNGs + JSON report."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mri_pairs_export_slice_compare import (  # noqa: E402
    load_by_kind,
    pair_slices_aligned,
    refine_mixed_nii_dicom_pair,
    to_uint8,
)

PWH_ROOT = Path(os.environ.get("ORALGPT_MRI_PWH_ROOT", "/path/to/PWH"))
OUT_DIR = Path(__file__).resolve().parent / "MRI_PWH_pair_audit"
REPORT_PATH = OUT_DIR / "pair_audit_report.json"
SUMMARY_PATH = OUT_DIR / "summary.md"

# NCC thresholds for training recommendation
NCC_INCLUDE = 0.55
NCC_MARGINAL = 0.35


def stem_from_nii(name: str) -> str:
    return name[: -len(".nii.gz")] if name.endswith(".nii.gz") else name


def match_t2fs_dir(stem: str) -> Path | None:
    key = stem.lower()
    for child in (PWH_ROOT / "t2fs").iterdir():
        if child.is_dir() and child.name.lower() == key:
            return child
    return None


def recommend_tier(ncc: float) -> str:
    if ncc >= NCC_INCLUDE:
        return "include"
    if ncc >= NCC_MARGINAL:
        return "marginal"
    return "exclude"


def save_audit_png(
    stem: str,
    *,
    alignment_mode: str,
    z_mid: float | None,
    idx_l: int,
    zl: float,
    idx_r: int,
    zr: float,
    pix_l: tuple[float, float],
    pix_r: tuple[float, float],
    sl_t1: np.ndarray,
    sl_t2: np.ndarray,
    plane_tag: str,
    plane_ncc: float,
    tier: str,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpl_dir = OUT_DIR / ".mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    Lu = to_uint8(sl_t1)
    Ru = to_uint8(sl_t2)
    fig_h = 4.5
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(fig_h * (Lu.shape[1] / max(Lu.shape[0], 1) + Ru.shape[1] / max(Ru.shape[0], 1)), fig_h),
    )
    axes[0].imshow(Lu, cmap="gray", aspect="equal", origin="upper")
    axes[0].axis("off")
    axes[0].set_title("T1W NIfTI", fontsize=9)
    axes[1].imshow(Ru, cmap="gray", aspect="equal", origin="upper")
    axes[1].axis("off")
    axes[1].set_title("t2fs DICOM", fontsize=9)

    mids = f"z_mid≈{z_mid:.2f}mm" if z_mid is not None else "z: fallback"
    suptitle = (
        f"PWH {stem}  |  recommend={tier}  |  plane NCC={plane_ncc:.3f}  ({plane_tag})\n"
        f"{alignment_mode}  |  {mids}  |  "
        f"T1 idx={idx_l} z={zl:.2f}  |  T2 idx={idx_r} z={zr:.2f}  |  "
        f"spacing mm: T1({pix_l[0]:.3f},{pix_l[1]:.3f}) T2({pix_r[0]:.3f},{pix_r[1]:.3f})"
    )
    fig.suptitle(suptitle, fontsize=8)
    out = OUT_DIR / f"PWH_{stem}_audit.png"
    plt.tight_layout(rect=[0.0, 0.04, 1.0, 0.92])
    plt.savefig(out, dpi=120, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out


def list_nii_cases() -> list[Path]:
    return sorted(
        [
            p
            for p in (PWH_ROOT / "T1W").glob("*.nii.gz")
            if p.is_file() and not p.name.startswith("._")
        ],
        key=lambda p: p.name.lower(),
    )


def audit_one(t1_path: Path) -> dict:
    stem = stem_from_nii(t1_path.name)
    t2_dir = match_t2fs_dir(stem)
    if t2_dir is None:
        return {"stem": stem, "status": "no_t2fs"}

    lv = load_by_kind(t1_path, "nii")
    rv = load_by_kind(t2_dir, "dicom")
    sl_t1, sl_t2, idx_l, idx_r, zl, zr, z_mid, mode = pair_slices_aligned(lv, rv)
    il0, ir0 = int(idx_l), int(idx_r)
    idx_l, idx_r, sl_t1, sl_t2, plane_tag, plane_ncc, refined = refine_mixed_nii_dicom_pair(
        lv, rv, idx_l, idx_r, nii_is_left=True, window=2
    )
    zl = float(lv.z_mm[idx_l])
    zr = float(rv.z_mm[idx_r])
    tier = recommend_tier(plane_ncc)

    png = save_audit_png(
        stem,
        alignment_mode=mode,
        z_mid=float(z_mid) if z_mid is not None else None,
        idx_l=int(idx_l),
        zl=zl,
        idx_r=int(idx_r),
        zr=zr,
        pix_l=(float(lv.spacing_y_mm), float(lv.spacing_x_mm)),
        pix_r=(float(rv.spacing_y_mm), float(rv.spacing_x_mm)),
        sl_t1=sl_t1,
        sl_t2=sl_t2,
        plane_tag=plane_tag,
        plane_ncc=float(plane_ncc),
        tier=tier,
    )

    return {
        "stem": stem,
        "status": "ok",
        "png": str(png),
        "t1_nii": str(t1_path),
        "t2fs_dir": str(t2_dir),
        "alignment_mode": mode,
        "z_overlap": mode == "physical_overlap",
        "physical_midpoint_z_mm": float(z_mid) if z_mid is not None else None,
        "t1_shape_nzyx": [int(x) for x in lv.data.shape],
        "t2_shape_nzyx": [int(x) for x in rv.data.shape],
        "t1_slice_index": int(idx_l),
        "t2_slice_index": int(idx_r),
        "slice_index_before_refine": [il0, ir0],
        "slice_pair_refined": bool(refined),
        "t1_z_mm": zl,
        "t2_z_mm": zr,
        "z_delta_mm": float(abs(zl - zr)),
        "plane_transform": plane_tag,
        "plane_ncc": round(float(plane_ncc), 4),
        "recommend_tier": tier,
        "recommend_training": tier == "include",
    }


def write_summary(report: dict) -> None:
    cases = report["cases"]
    ok = [c for c in cases if c.get("status") == "ok"]
    by_tier = {t: [c["stem"] for c in ok if c["recommend_tier"] == t] for t in ("include", "marginal", "exclude")}

    lines = [
        "# PWH T1W (NIfTI) vs t2fs (DICOM) audit",
        "",
        f"- Total NIfTI cases: {report['total_nii']}",
        f"- Audited OK: {len(ok)}",
        f"- Recommend **include** (NCC≥{NCC_INCLUDE}): {len(by_tier['include'])}",
        f"- **marginal** ({NCC_MARGINAL}≤NCC<{NCC_INCLUDE}): {len(by_tier['marginal'])}",
        f"- **exclude** (NCC<{NCC_MARGINAL}): {len(by_tier['exclude'])}",
        "",
        "## Plane transform counts",
        "",
    ]
    from collections import Counter

    tc = Counter(c["plane_transform"] for c in ok)
    for tag, n in tc.most_common():
        lines.append(f"- `{tag}`: {n}")

    lines.extend(["", "## Exclude (poor NCC)", ""])
    for s in sorted(by_tier["exclude"]):
        c = next(x for x in ok if x["stem"] == s)
        lines.append(f"- {s}: NCC={c['plane_ncc']}, plane={c['plane_transform']}")

    lines.extend(["", "## Marginal (review manually)", ""])
    for s in sorted(by_tier["marginal"], key=lambda x: next(
        c["plane_ncc"] for c in ok if c["stem"] == x
    )):
        c = next(x for x in ok if x["stem"] == s)
        lines.append(
            f"- {s}: NCC={c['plane_ncc']}, plane={c['plane_transform']}, "
            f"shapes T1{c['t1_shape_nzyx']} T2{c['t2_shape_nzyx']}"
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    nii_list = list_nii_cases()
    cases: list[dict] = []
    for i, t1_path in enumerate(nii_list):
        try:
            row = audit_one(t1_path)
        except Exception as exc:
            row = {
                "stem": stem_from_nii(t1_path.name),
                "status": "error",
                "error": str(exc),
            }
        cases.append(row)
        if row.get("status") == "ok":
            print(
                f"[{i+1}/{len(nii_list)}] {row['stem']}: "
                f"tier={row['recommend_tier']} ncc={row['plane_ncc']} "
                f"plane={row['plane_transform']}",
                flush=True,
            )
        else:
            print(f"[{i+1}/{len(nii_list)}] {row['stem']}: {row['status']}", flush=True)

    report = {
        "cohort": "PWH",
        "data_root": str(PWH_ROOT),
        "pairing": "T1W/*.nii.gz <-> t2fs/{stem}/ (case-insensitive)",
        "ncc_thresholds": {
            "include": f">= {NCC_INCLUDE}",
            "marginal": f"{NCC_MARGINAL} - {NCC_INCLUDE}",
            "exclude": f"< {NCC_MARGINAL}",
        },
        "note": (
            "plane_ncc is 2-D normalized cross-correlation after heuristic "
            "NIfTI plane remap (not 3-D registration). z uses physical overlap."
        ),
        "total_nii": len(nii_list),
        "cases": cases,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_summary(report)
    print(f"\nWrote {len(cases)} entries to {REPORT_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
