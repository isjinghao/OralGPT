"""Heuristic 2-D remap for NiFTI vs DICOM previews (transpose / rot90 / flips)."""

from __future__ import annotations

import os

import numpy as np
from scipy.ndimage import zoom


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64).ravel()
    bb = b.astype(np.float64).ravel()
    aa -= aa.mean()
    bb -= bb.mean()
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-9)
    return float(np.dot(aa, bb) / den)


def _pn(x: np.ndarray) -> np.ndarray:
    xp = x.astype(np.float64)
    lo, hi = np.percentile(xp, [5.0, 95.0])
    xp = np.clip(xp, lo, hi)
    return ((xp - lo) / (hi - lo + 1e-9)).astype(np.float32)


def _resize(x: np.ndarray, sh: tuple[int, int]) -> np.ndarray:
    fy = sh[0] / x.shape[0]
    fx = sh[1] / x.shape[1]
    return zoom(x, (fy, fx), order=1)


def plane_transform_candidates() -> list[tuple[str, object]]:
    base: list[tuple[str, object]] = [
        ("identity", lambda x: x),
        ("transpose", lambda x: np.ascontiguousarray(x.T)),
        ("rot90:+1", lambda x: np.ascontiguousarray(np.rot90(x, k=1))),
        ("rot90:+3", lambda x: np.ascontiguousarray(np.rot90(x, k=3))),
        ("transpose+fliplr", lambda x: np.ascontiguousarray(np.fliplr(np.ascontiguousarray(x.T)))),
        ("transpose+flipud", lambda x: np.ascontiguousarray(np.flipud(np.ascontiguousarray(x.T)))),
        ("transpose+rot90:+1", lambda x: np.ascontiguousarray(np.rot90(np.ascontiguousarray(x.T), k=1))),
        ("transpose+rot90:+3", lambda x: np.ascontiguousarray(np.rot90(np.ascontiguousarray(x.T), k=3))),
    ]

    cand: list[tuple[str, object]] = []
    for tag, fn in base:
        cand.append((tag, fn))
        cand.append(
            (
                tag + "+flipud",
                lambda x, f=fn: np.ascontiguousarray(
                    np.flipud(np.asarray(f(x), dtype=np.float32))
                ),
            )
        )

    return cand


def apply_plane_transform(x: np.ndarray, transform_name: str) -> np.ndarray:
    for tag, fn in plane_transform_candidates():
        if tag == transform_name:
            return np.ascontiguousarray(np.asarray(fn(x), dtype=np.float32))
    raise KeyError(f"Unknown plane transform: {transform_name}")


def score_plane_transforms(
    moving: np.ndarray, dicom_reference: np.ndarray
) -> list[tuple[str, float]]:
    bench = _resize(_pn(np.ascontiguousarray(dicom_reference.astype(np.float32))), (176, 176))
    mm0 = np.ascontiguousarray(moving.astype(np.float32))
    scores: list[tuple[str, float]] = []

    for tag, fn in plane_transform_candidates():
        try:
            yt = np.asarray(fn(mm0), dtype=np.float32)
            if yt.ndim != 2:
                continue
            tst = _resize(_pn(yt), bench.shape)
            sc = _ncc(bench, tst)
        except Exception:
            continue
        scores.append((tag, float(sc)))

    return sorted(scores, key=lambda item: item[1], reverse=True)


def align_moving_plane_to_dicom_preview(
    moving: np.ndarray, dicom_reference: np.ndarray
) -> tuple[np.ndarray, str, float]:
    scores = score_plane_transforms(moving, dicom_reference)
    if not scores:
        out = np.ascontiguousarray(moving.astype(np.float32))
        return out, "identity", float("-inf")

    name_best, score_best = scores[0]
    out = apply_plane_transform(moving, name_best)
    return out, name_best, float(score_best)

