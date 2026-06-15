#!/usr/bin/env python3
"""
Per-cohort T1*/T2FS QA: paired 2-D slices from overlap along physical slice direction (RAS
slice-normal projections; DICOM IPP + IOP-derived normal—not Z-only heuristic). Mixed
NiFTI+DICOM: ±2-layer index search plus in-plane remap (preview only). PWH: one legacy
entry (s006) plus configurable extra *.nii.gz subjects vs matched t2fs folders. Writes PNGs +
latest_pair_preview_meta.json under OralGPT-X/MRI_pair_slice_examples/.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pydicom  # noqa: E402
from nibabel.affines import apply_affine  # noqa: E402
from nibabel import as_closest_canonical  # noqa: E402
from nibabel.loadsave import load as nib_load  # noqa: E402

from mri_visual_plane_align import align_moving_plane_to_dicom_preview  # noqa: E402

BASE_OUT = Path(os.environ.get("ORALGPT_X_WORKDIR", str(Path.cwd())))
OUT_DIR = BASE_OUT / "MRI_pair_slice_examples"
PAIR_META_PATH = OUT_DIR / "latest_pair_preview_meta.json"

PWH_ROOT = Path(os.environ.get("ORALGPT_MRI_PWH_ROOT", "/path/to/PWH"))
# NIfTI T1W vs t2fs DICOM: besides the fixed s006 entry, generate this many more subjects.
PWH_PREVIEW_EXTRA_CASES = 12


def _stem_from_nii_gz_filename(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    raise ValueError(name)


def _pwh_matching_t2fs_dir(stem: str) -> Path | None:
    key = stem.lower()
    root = PWH_ROOT / "t2fs"
    if not root.is_dir():
        return None
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == key:
            return child
    return None


def _build_pwh_pair_specs(extra_cases: int) -> list[dict]:
    """PWH previews: canonical s006 (legacy PNG basename) plus up to ``extra_cases`` others."""
    t1w = PWH_ROOT / "T1W"
    out: list[dict] = []
    if not t1w.is_dir():
        return out

    nii_list = sorted(
        [p for p in t1w.glob("*.nii.gz") if p.is_file()],
        key=lambda p: p.name.lower(),
    )
    if not nii_list:
        return out

    for path_s006 in nii_list:
        stem_m = _stem_from_nii_gz_filename(path_s006.name)
        if stem_m.lower() != "s006":
            continue
        t2 = _pwh_matching_t2fs_dir(stem_m)
        if t2 is None:
            break
        out.append(
            {
                "name": "02_PWH_T1W_vs_t2fs",
                "left_path": path_s006,
                "right_path": t2,
                "kind": ("nii", "dicom"),
            }
        )
        break

    extras = 0
    for p in nii_list:
        stem = _stem_from_nii_gz_filename(p.name)
        if stem.lower() == "s006":
            continue
        if extras >= extra_cases:
            break
        t2 = _pwh_matching_t2fs_dir(stem)
        if t2 is None:
            continue
        out.append(
            {
                "name": f"02_PWH_{stem}_T1W_vs_t2fs",
                "left_path": p,
                "right_path": t2,
                "kind": ("nii", "dicom"),
            }
        )
        extras += 1

    return out


_SPEC_GUIZ = {
    "name": "01_Guizhou_T1WI_vs_T2FS",
    "left_path": Path(
        os.environ.get("ORALGPT_MRI_GUIZHOU_T1_EXAMPLE", "/path/to/Guizhou/T1WI/01")
    ),
    "right_path": Path(
        os.environ.get("ORALGPT_MRI_GUIZHOU_T2_EXAMPLE", "/path/to/Guizhou/T2FS/01")
    ),
    "kind": ("dicom", "dicom"),
}
_SPEC_PEKING = {
    "name": "03_Peking_T1_vs_T2FS",
    "left_path": Path(
        os.environ.get("ORALGPT_MRI_PEKING_ROOT", "/path/to/Peking") + "/"
        "T1/001/4a8feea5/19a93d35"
    ),
    "right_path": Path(
        os.environ.get("ORALGPT_MRI_PEKING_ROOT", "/path/to/Peking") + "/"
        "T2FS/001/19a93cb6"
    ),
    "kind": ("dicom", "dicom"),
}
_SPEC_KWC = {
    "name": "04_KWC_T1WI_vs_T2FS",
    "left_path": Path(
        os.environ.get("ORALGPT_MRI_KWC_T1_EXAMPLE", "/path/to/KWC/T1WI/03")
    ),
    "right_path": Path(
        os.environ.get("ORALGPT_MRI_KWC_T2_EXAMPLE", "/path/to/KWC/T2FS/03")
    ),
    "kind": ("dicom", "dicom"),
}

PAIR_SPECS: list[dict] = [
    _SPEC_GUIZ,
    *_build_pwh_pair_specs(PWH_PREVIEW_EXTRA_CASES),
    _SPEC_PEKING,
    _SPEC_KWC,
]


@dataclass(frozen=True)
class LoadedVolume:
    data: np.ndarray  # float32 (Nz, Ny, Nx); stack dimension axis 0
    z_mm: np.ndarray  # (Nz,)
    spacing_y_mm: float
    spacing_x_mm: float


def _iter_candidate_files(root: Path) -> Iterable[Path]:
    skip = re.compile(r"^(?:\._|\.DS_Store$|DIRFILE)", re.I)
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if skip.match(fn):
                continue
            if "seg" in fn.lower():
                continue
            yield Path(dp) / fn


def _dicom_physical_z(ds: pydicom.dataset.FileDataset) -> float | None:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        z = float(ipp[2])
        if np.isfinite(z):
            return z
    sl = float(getattr(ds, "SliceLocation", np.nan))
    if np.isfinite(sl):
        return sl
    return None


def _ipp_lps_to_ras_position(ipp) -> np.ndarray:
    """DICOM patient (LPS) → nibabel-style RAS position (mm)."""
    return np.array(
        [-float(ipp[0]), -float(ipp[1]), float(ipp[2])], dtype=np.float64
    )


def _iop_slice_normal_lps_to_ras(iop) -> np.ndarray | None:
    """Unit slice normal in RAS from ImageOrientationPatient (row, col in LPS)."""
    r = np.array([float(iop[0]), float(iop[1]), float(iop[2])], dtype=np.float64)
    c = np.array([float(iop[3]), float(iop[4]), float(iop[5])], dtype=np.float64)
    n_lps = np.cross(r, c)
    norm = float(np.linalg.norm(n_lps))
    if norm < 1e-9:
        return None
    n_lps = n_lps / norm
    return np.array([-n_lps[0], -n_lps[1], n_lps[2]], dtype=np.float64)


def _dicom_slice_coord_mm_sorted(
    items: list[tuple[np.ndarray, pydicom.dataset.FileDataset]],
) -> np.ndarray | None:
    """Comparable scalar position per slice along stack normal (RAS frame). Fallback None."""
    if not items:
        return None

    ds0 = items[0][1]
    iop = getattr(ds0, "ImageOrientationPatient", None)
    if iop is None or len(iop) < 6:
        return None

    n_ras = _iop_slice_normal_lps_to_ras(iop)
    if n_ras is None:
        return None

    ipp0 = getattr(ds0, "ImagePositionPatient", None)
    ippn = getattr(items[-1][1], "ImagePositionPatient", None)
    if ipp0 is None or len(ipp0) < 3 or ippn is None or len(ippn) < 3:
        return None

    p0 = _ipp_lps_to_ras_position(ipp0)
    p_last = _ipp_lps_to_ras_position(ippn)
    if float(np.dot(p_last - p0, n_ras)) < 0.0:
        n_ras = -n_ras

    out: list[float] = []
    for _a, ds in items:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is None or len(ipp) < 3:
            out.append(np.nan)
            continue
        p = _ipp_lps_to_ras_position(ipp)
        out.append(float(np.dot(p, n_ras)))

    return np.asarray(out, dtype=np.float64)


def _dicom_sort_key(ds: pydicom.dataset.FileDataset) -> tuple[float, ...]:
    z = _dicom_physical_z(ds)
    sl = float(getattr(ds, "SliceLocation", np.nan))
    ipp_z = (
        float(getattr(ds, "ImagePositionPatient", [np.nan, np.nan, np.nan])[2])
        if getattr(ds, "ImagePositionPatient", None) is not None
        else np.nan
    )
    inst = int(getattr(ds, "InstanceNumber", 0))
    zn = np.nan if z is None else float(z)
    return (zn, sl, ipp_z, inst)


def load_dicom_volume_bundle(root: Path) -> LoadedVolume:
    items: list[tuple[np.ndarray, pydicom.dataset.FileDataset]] = []
    spacing_y_mm = float("nan")
    spacing_x_mm = float("nan")

    for fp in _iter_candidate_files(root):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                ds = pydicom.dcmread(str(fp), stop_before_pixels=False, force=True)
        except Exception:
            continue

        if getattr(ds, "SamplesPerPixel", 1) != 1:
            continue
        if getattr(ds, "NumberOfFrames", 1) not in (1, None):
            continue
        try:
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            continue
        if arr.ndim != 2:
            continue

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        spacing = getattr(ds, "PixelSpacing", None)
        if spacing is not None:
            fy, fx = float(spacing[0]), float(spacing[1])
            if np.isfinite(fy):
                spacing_y_mm, spacing_x_mm = fy, fx

        items.append((arr, ds))

    if not items:
        raise RuntimeError(f"No readable DICOM slices under {root}")

    items.sort(key=lambda t: _dicom_sort_key(t[1]))

    planes = [np.ascontiguousarray(arr, dtype=np.float32) for arr, _ in items]
    sh0 = tuple(int(x) for x in planes[0].shape)
    if any(p.shape != sh0 for p in planes):
        shapes = sorted({tuple(p.shape) for p in planes})
        raise RuntimeError(
            f"Inconsistent in-plane shapes under {root}: {shapes[:5]} ..."
        )

    z_geo = _dicom_slice_coord_mm_sorted(items)

    zcoords_raw: list[float | None] = []
    surrogate_warned = False

    for _arr, ds in items:
        z = _dicom_physical_z(ds)
        zcoords_raw.append(float(z) if z is not None else None)

    nz = len(zcoords_raw)
    finite_pts = np.array([float(v) for v in zcoords_raw if v is not None], dtype=float)

    dz0 = np.nanmedian(np.diff(np.sort(finite_pts))) if finite_pts.size >= 2 else np.nan

    dz = float(abs(dz0)) if np.isfinite(float(dz0)) else 1.0
    fallback_base = float(zcoords_raw[0]) if zcoords_raw[0] is not None else 0.0

    zs_f: list[float] = []
    for i in range(nz):
        if z_geo is not None and np.isfinite(z_geo[i]):
            zs_f.append(float(z_geo[i]))
            continue

        z = zcoords_raw[i]
        if z is not None:
            zs_f.append(float(z))
            continue

        zs_f.append((zs_f[-1] + dz) if zs_f else (fallback_base + float(i) * dz))

        if not surrogate_warned:
            surrogate_warned = True
            warnings.warn(
                f"Some slices under {root} lack IPP/SliceLocation; inferring surrogate z "
                f"with spacing ~= {dz:.3f} mm.",
                RuntimeWarning,
            )

    zs_arr = np.asarray(zs_f, dtype=np.float64)

    vol = np.stack(planes, axis=0).astype(np.float32)

    return LoadedVolume(
        data=vol,
        z_mm=np.asarray(zs_arr, dtype=np.float32),
        spacing_y_mm=spacing_y_mm,
        spacing_x_mm=spacing_x_mm,
    )


def _infer_slice_axis(shape: tuple[int, int, int], voxel_mm: tuple[float, float, float]) -> int:
    extent = tuple(float(shape[i]) * voxel_mm[i] for i in range(3))
    return int(np.argmin(extent))


def load_nifti_bundle(path: Path) -> LoadedVolume:
    img = as_closest_canonical(nib_load(str(path)))
    data = img.get_fdata(dtype=np.float32)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise RuntimeError(f"Expected 3-D NIfTI, got shape {data.shape} for {path}")

    voxel = tuple(float(img.header.get_zooms()[i]) for i in range(3))
    ax = _infer_slice_axis(tuple(int(s) for s in data.shape), voxel)

    planes: list[np.ndarray] = []
    affine = img.affine
    z_mm_vals: list[float] = []

    mid = [(data.shape[d] - 1) / 2.0 for d in range(3)]

    nz = int(data.shape[ax])
    if nz < 2:
        raise RuntimeError(f"Degenerate slice count along axis {ax} for {path}")

    M = affine[:3, :3]
    step = np.zeros(3, dtype=np.float64)
    step[ax] = 1.0
    n_raw = M @ step
    n_norm = float(np.linalg.norm(n_raw))
    if n_norm < 1e-9:
        n_ras = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n_ras = (n_raw / n_norm).astype(np.float64)

    idx0 = np.asarray(mid[:], dtype=np.float64)
    idx0[ax] = 0.0
    idx1 = np.asarray(mid[:], dtype=np.float64)
    idx1[ax] = float(nz - 1)
    p0v = np.asarray(apply_affine(affine, idx0[:3])[:3], dtype=np.float64)
    p1v = np.asarray(apply_affine(affine, idx1[:3])[:3], dtype=np.float64)
    if float(np.dot(p1v - p0v, n_ras)) < 0.0:
        n_ras = -n_ras

    for k in range(nz):
        idx = np.asarray(mid[:], dtype=np.float64)
        idx[ax] = float(k)
        pv = np.asarray(apply_affine(affine, idx[:3])[:3], dtype=np.float64)

        planes.append(np.ascontiguousarray(np.take(data, k, axis=ax)).astype(np.float32))
        z_mm_vals.append(float(np.dot(pv, n_ras)))

    z_arr = np.asarray(z_mm_vals, dtype=np.float32)

    fy = float(abs(voxel[(ax + 1) % 3]))
    fx = float(abs(voxel[(ax + 2) % 3]))

    vol = np.stack(planes, axis=0)

    return LoadedVolume(data=vol, z_mm=z_arr, spacing_y_mm=fy, spacing_x_mm=fx)


def load_by_kind(path: Path, kind: Literal["nii", "dicom"]) -> LoadedVolume:
    if kind == "nii":
        return load_nifti_bundle(path)
    root = path if path.is_dir() else path.parent
    return load_dicom_volume_bundle(root)


def _physical_overlap_target(zl: np.ndarray, zr: np.ndarray) -> tuple[float | None, str]:
    """Return shared-space midpoint IF intervals overlap materially."""
    lo = float(max(np.nanmin(zl), np.nanmin(zr)))
    hi = float(min(np.nanmax(zl), np.nanmax(zr)))
    min_span = float(
        max(
            1e-3,
            min(float(np.nanmax(zl) - np.nanmin(zl)), float(np.nanmax(zr) - np.nanmin(zr)))
            / 256.0,
        )
    )
    if np.isfinite(lo) and np.isfinite(hi) and hi - lo > min_span:
        return 0.5 * (lo + hi), "physical_overlap"
    return None, "no_physical_overlap"


def pick_slice_near_target(
    vol: LoadedVolume, target_mm: float
) -> tuple[np.ndarray, int, float]:
    z = np.asarray(vol.z_mm, dtype=np.float64)
    idx = int(np.argmin(np.abs(z - float(target_mm))))
    return vol.data[idx, ...].astype(np.float32), idx, float(z[idx])


def pick_slice_at_relative_fraction(
    vol: LoadedVolume, frac: float
) -> tuple[np.ndarray, int, float, float]:
    z = np.asarray(vol.z_mm, dtype=np.float64)
    denom = float(z.max() - z.min())
    if not np.isfinite(denom) or denom < 1e-9:
        idx = max(0, min(int(vol.data.shape[0]) - 1, int(vol.data.shape[0] // 2)))
        return vol.data[idx, ...].astype(np.float32), idx, float(z[idx]), float(idx)
    z_tgt = float(z.min()) + float(frac) * denom
    return (*pick_slice_near_target(vol, z_tgt), z_tgt)


def pair_slices_aligned(
    lv: LoadedVolume, rv: LoadedVolume
) -> tuple[
    np.ndarray,
    np.ndarray,
    int,
    int,
    float,
    float,
    float | None,
    str,
]:
    """Return paired slices plus indices and alignment mode."""

    tz, tag = _physical_overlap_target(lv.z_mm, rv.z_mm)
    if tz is None:
        frac = 0.5
        mode = "fallback_relative_depth_frac=0.5"
        sl_left, idx_l, zl_sel, _ = pick_slice_at_relative_fraction(lv, frac)
        sl_right, idx_r, zr_sel, _ = pick_slice_at_relative_fraction(rv, frac)
        return (
            sl_left,
            sl_right,
            idx_l,
            idx_r,
            float(zl_sel),
            float(zr_sel),
            None,
            mode,
        )

    sl_left, idx_l, zl_sel = pick_slice_near_target(lv, tz)
    sl_right, idx_r, zr_sel = pick_slice_near_target(rv, tz)
    return (
        sl_left,
        sl_right,
        idx_l,
        idx_r,
        float(zl_sel),
        float(zr_sel),
        float(tz),
        tag,
    )


def refine_mixed_nii_dicom_pair(
    lv: LoadedVolume,
    rv: LoadedVolume,
    idx_l: int,
    idx_r: int,
    *,
    nii_is_left: bool,
    window: int = 2,
    max_pair_z_delta_mm: float = 4.0,
) -> tuple[int, int, np.ndarray, np.ndarray, str, float, bool]:
    """Local ±window stack search maximizing the same preview-plane NCC."""
    nl = int(lv.data.shape[0])
    nr = int(rv.data.shape[0])
    w = max(0, int(window))
    ztol = float(max_pair_z_delta_mm)

    def run_search(z_gate: float | None) -> tuple[float, int, int, np.ndarray, np.ndarray, str]:
        best_sc = -1e18
        best_il, best_ir = int(idx_l), int(idx_r)
        best_sl = np.ascontiguousarray(lv.data[best_il].astype(np.float32))
        best_sr = np.ascontiguousarray(rv.data[best_ir].astype(np.float32))
        best_tag = "identity"

        for il in range(max(0, idx_l - w), min(nl, idx_l + w + 1)):
            for ir in range(max(0, idx_r - w), min(nr, idx_r + w + 1)):
                if z_gate is not None:
                    zd = abs(float(lv.z_mm[il]) - float(rv.z_mm[ir]))
                    if zd > z_gate:
                        continue
                sl = lv.data[il].astype(np.float32)
                sr = rv.data[ir].astype(np.float32)
                if nii_is_left:
                    mv_al, tag, sc = align_moving_plane_to_dicom_preview(sl, sr)
                    if sc > best_sc:
                        best_sc = sc
                        best_il, best_ir = il, ir
                        best_sl = mv_al
                        best_sr = sr
                        best_tag = tag
                else:
                    mv_al, tag, sc = align_moving_plane_to_dicom_preview(sr, sl)
                    if sc > best_sc:
                        best_sc = sc
                        best_il, best_ir = il, ir
                        best_sl = sl
                        best_sr = mv_al
                        best_tag = tag

        return best_sc, best_il, best_ir, best_sl, best_sr, best_tag

    sc0, il0, ir0, sl0, sr0, tag0 = run_search(ztol)
    if sc0 < -1e17:
        sc0, il0, ir0, sl0, sr0, tag0 = run_search(None)

    refined = (il0 != idx_l) or (ir0 != idx_r)

    return il0, ir0, sl0, sr0, tag0, float(sc0), refined


def to_uint8(slice2d: np.ndarray) -> np.ndarray:
    x = slice2d.astype(np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(slice2d, dtype=np.uint8)
    lo, hi = np.percentile(finite, (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return (x * 255.0).astype(np.uint8)


def save_pair_png(
    name: str,
    title: str,
    alignment_mode: str,
    shared_physical_midpoint_z_mm: float | None,
    idx_l: int,
    zl_use: float,
    idx_r: int,
    zr_use: float,
    pix_l: tuple[float, float],
    pix_r: tuple[float, float],
    sl_left: np.ndarray,
    sl_right: np.ndarray,
    mixed_nii_dicom: bool,
    plane_view_fix: str | None = None,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpl_dir = OUT_DIR / ".mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    Lu = to_uint8(sl_left)
    Ru = to_uint8(sl_right)

    fig_h = 5.0
    fig, axes = plt.subplots(1, 2, figsize=(fig_h * (Lu.shape[1] / Lu.shape[0] + Ru.shape[1] / Ru.shape[0]), fig_h))

    axes[0].imshow(Lu, cmap="gray", aspect="equal", origin="upper")
    axes[0].axis("off")
    axes[0].set_title("T1* (native pixels)", fontsize=10)

    axes[1].imshow(Ru, cmap="gray", aspect="equal", origin="upper")
    axes[1].axis("off")
    axes[1].set_title("T2FS (native pixels)", fontsize=10)

    sy_l, sx_l = pix_l
    sy_r, sx_r = pix_r

    note_lines: list[str] = []
    if alignment_mode.startswith("fallback"):
        note_lines.append(
            "note: no trustworthy shared physical z-interval (or RAS/LPS differs); "
            "using relative slab-depth fraction (0.5)."
        )

    if plane_view_fix:
        note_lines.append(plane_view_fix)
    elif mixed_nii_dicom:
        note_lines.append(
            "note (NiFTI+DICOM): mixed modalities preview without heuristic plane remap."
        )


    mids = ""
    if shared_physical_midpoint_z_mm is not None:
        mids = f"\nphysical midpoint z_tgt ≈ {shared_physical_midpoint_z_mm:.3f} mm"

    suptitle = (
        title
        + f"\nalign: {alignment_mode}{mids}"
        + f"\nL idx={idx_l} z={zl_use:.3f}  ΔyΔx(mm)=({sy_l:.3f},{sx_l:.3f})"
        + f"\nR idx={idx_r} z={zr_use:.3f}  ΔyΔx(mm)=({sy_r:.3f},{sx_r:.3f})"
    )

    if note_lines:
        suptitle += "\n" + "\n".join(note_lines)

    fig.suptitle(suptitle, fontsize=9)
    outp = OUT_DIR / f"{name}_zaligned_slice_compare.png"

    plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.93])
    plt.savefig(outp, dpi=144, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    return outp


def main() -> None:
    meta: dict[str, dict] = {}

    try:
        pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        from pydicom import config as _cfg

        _cfg.settings.encoding = None
    except Exception:
        pass

    for spec in PAIR_SPECS:
        lk, rk = spec["kind"]

        lv = load_by_kind(Path(spec["left_path"]), lk)  # type: ignore[arg-type]
        rv = load_by_kind(Path(spec["right_path"]), rk)  # type: ignore[arg-type]

        (
            sl_left,
            sl_right,
            idx_l,
            idx_r,
            zl_sel,
            zr_sel,
            z_mid_physical,
            mode,
        ) = pair_slices_aligned(lv, rv)

        pix_l = (lv.spacing_y_mm, lv.spacing_x_mm)
        pix_r = (rv.spacing_y_mm, rv.spacing_x_mm)

        short_l = str(spec["left_path"]).split("/SGT/", 1)[-1]
        short_r = str(spec["right_path"]).split("/SGT/", 1)[-1]

        plane_note: str | None = None
        plane_view_tag: str | None = None
        plane_view_ncc: float | None = None
        slice_pair_refined = False
        idx_before_refine: tuple[int, int] | None = None

        if lk != rk:
            idx_before_refine = (int(idx_l), int(idx_r))
            idx_l, idx_r, sl_left, sl_right, plane_view_tag, plane_view_ncc, slice_pair_refined = (
                refine_mixed_nii_dicom_pair(
                    lv,
                    rv,
                    idx_l,
                    idx_r,
                    nii_is_left=(lk == "nii"),
                    window=2,
                )
            )
            zl_sel = float(lv.z_mm[idx_l])
            zr_sel = float(rv.z_mm[idx_r])
            plane_note = (
                f"NiFTI+DICOM preview: {plane_view_tag} "
                f"(176² NCC≈{plane_view_ncc:.3f}); "
                "slice ±2 with |Δz|≤4 mm before NCC"
                + ("; indices adjusted" if slice_pair_refined else "; kept overlap init")
                + "; not 3-D registration"
            )

        outp = save_pair_png(
            spec["name"],
            spec["name"].replace("_", " ")
            + f"\nL: …/SGT/{short_l}"
            + f"\nR: …/SGT/{short_r}",
            mode,
            float(z_mid_physical) if z_mid_physical is not None else None,
            idx_l,
            zl_sel,
            idx_r,
            zr_sel,
            pix_l,
            pix_r,
            sl_left,
            sl_right,
            lk != rk,
            plane_view_fix=plane_note,
        )

        meta[spec["name"]] = {
            "png": str(outp),
            "alignment_mode": mode,
            "plane_view_transform_preview": plane_view_tag,
            "plane_view_ncc_preview": plane_view_ncc,
            "plane_view_fix_text": plane_note,
            "mixed_slice_pair_refined": slice_pair_refined,
            "slice_idx_before_refine_if_mixed": list(idx_before_refine)
            if idx_before_refine is not None
            else None,
            "physical_midpoint_z_mm_if_used": (
                float(z_mid_physical) if z_mid_physical is not None else None
            ),
            "left_vol_shape_xyz": tuple(int(x) for x in lv.data.shape),
            "right_vol_shape_xyz": tuple(int(x) for x in rv.data.shape),
            "left_slice_index_axis0": idx_l,
            "right_slice_index_axis0": idx_r,
            "left_z_phys_or_surrogate_mm": float(zl_sel),
            "right_z_phys_or_surrogate_mm": float(zr_sel),
            "left_spacing_yx_mm": (float(pix_l[0]), float(pix_l[1])),
            "right_spacing_yx_mm": (float(pix_r[0]), float(pix_r[1])),
        }

        print("Wrote", outp)

    PAIR_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PAIR_META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Meta:", PAIR_META_PATH)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    main()
