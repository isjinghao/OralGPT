#!/usr/bin/env python3
"""Pixel-level metrics for edit benchmarks with paired ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as skimage_ssim
from tqdm import tqdm

from metrics.io_utils import append_jsonl, load_completed_ids
from path_utils import bench_root, resolve_path

try:
    import lpips  # type: ignore

    _LPIPS = lpips.LPIPS(net="alex")
    _HAS_LPIPS = True
except Exception:
    _LPIPS = None
    _HAS_LPIPS = False


def load_samples(metadata_file: Path) -> list[dict]:
    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload["samples"]
    return payload


def to_gray_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def resize_like(source: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(source.astype(np.uint8), mode="L")
    if image.size[::-1] != shape:
        image = image.resize((shape[1], shape[0]), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32)


def normalized_mutual_information(source: np.ndarray, target: np.ndarray) -> float:
    source_u8 = np.clip(source, 0, 255).astype(np.uint8)
    target_u8 = np.clip(target, 0, 255).astype(np.uint8)
    hist_2d, _, _ = np.histogram2d(source_u8.ravel(), target_u8.ravel(), bins=256, range=[[0, 256], [0, 256]])
    pxy = hist_2d / max(float(hist_2d.sum()), 1.0)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = px[:, None] * py[None, :]
    nz = pxy > 0
    mi = np.sum(pxy[nz] * np.log(pxy[nz] / np.maximum(px_py[nz], 1e-12)))
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    return float(2 * mi / max(hx + hy, 1e-12))


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(20 * np.log10(255.0) - 10 * np.log10(mse))


def compute_lpips(pred: np.ndarray, target: np.ndarray) -> float | None:
    if not _HAS_LPIPS:
        return None
    import torch

    pred_t = torch.from_numpy(pred / 127.5 - 1.0)[None, None, :, :]
    target_t = torch.from_numpy(target / 127.5 - 1.0)[None, None, :, :]
    with torch.no_grad():
        return float(_LPIPS(pred_t, target_t).item())


def score_pair(pred_path: Path, target_path: Path) -> dict[str, float | None]:
    pred = to_gray_array(pred_path)
    target = to_gray_array(target_path)
    pred_resized = resize_like(pred, target.shape)
    return {
        "ssim": float(skimage_ssim(pred_resized, target, data_range=255.0)),
        "psnr": psnr(pred_resized, target),
        "nmi": normalized_mutual_information(pred_resized, target),
        "mae": float(np.mean(np.abs(pred_resized - target))),
        "lpips": compute_lpips(pred_resized, target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_file", type=Path, required=True)
    parser.add_argument("--bench_data_root", type=Path, required=True)
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--benchmark", type=str, default="cbct")
    args = parser.parse_args()

    metadata_file = resolve_path(args.metadata_file, bench_root())
    bench_data_root = resolve_path(args.bench_data_root)
    pred_dir = resolve_path(args.pred_dir)
    output_jsonl = resolve_path(args.output_jsonl)

    samples = load_samples(metadata_file)
    completed = load_completed_ids(output_jsonl)

    for sample in tqdm(samples, desc="pixel-metrics"):
        sample_id = sample["id"]
        if sample_id in completed:
            continue
        task_type = sample.get("task_type", "default")
        pred_path = pred_dir / task_type / f"{sample_id}.png"
        target_path = bench_data_root / sample["target"]["image_path"]
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing prediction: {pred_path}")
        if not target_path.is_file():
            raise FileNotFoundError(f"Missing target: {target_path}")

        metrics = score_pair(pred_path, target_path)
        meta = sample.get("metadata", {})
        row = {
            "id": sample_id,
            "benchmark": args.benchmark,
            "task_type": task_type,
            **metrics,
        }
        for key in ("volume_id", "modality", "batch", "cohort", "case_id"):
            value = meta.get(key)
            if value is not None:
                row[key] = value
        append_jsonl(output_jsonl, row)

    print(f"Wrote pixel metrics to {output_jsonl}")


if __name__ == "__main__":
    main()
