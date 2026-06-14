#!/usr/bin/env python3
"""Shared CBCT preprocessing helpers for low-dose to standard-dose pairing.

The BAGEL dataset builders import ``dicom_to_preprocessed_uint8`` directly so
the same intensity preprocessing is applied before train/test parquet creation.
The CLI is kept for quick PNG export and visual inspection of the preprocessed
DICOM slices.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_INPUT = None
DEFAULT_OUTPUT = Path.cwd() / "preprocessed_png"
DEFAULT_CLIP_MIN = -1000.0
DEFAULT_CLIP_MAX = 3000.0


def as_float(value: object, default: float | None = None) -> float | None:
    try:
        if not isinstance(value, (str, bytes)) and hasattr(value, "__iter__"):
            value = next(iter(value))
        return float(value)
    except Exception:
        return default


def preprocess_numpy(
    image: np.ndarray,
    clip_min: float = DEFAULT_CLIP_MIN,
    clip_max: float = DEFAULT_CLIP_MAX,
) -> np.ndarray:
    """Clip CBCT intensities and rescale one slice/volume to [0, 1]."""
    if clip_max <= clip_min:
        raise ValueError("clip_max must be larger than clip_min")

    image = image.astype(np.float32, copy=False)
    image = np.clip(image, clip_min, clip_max)
    return (image - clip_min) / (clip_max - clip_min)


def dicom_to_preprocessed_uint8(
    ds: object,
    pixels: np.ndarray | None = None,
    clip_min: float = DEFAULT_CLIP_MIN,
    clip_max: float = DEFAULT_CLIP_MAX,
) -> np.ndarray:
    """Apply DICOM rescale metadata, clip intensities, and return uint8 pixels."""
    if pixels is None:
        pixels = ds.pixel_array

    image = pixels.astype(np.float32)
    slope = as_float(getattr(ds, "RescaleSlope", 1.0), 1.0) or 1.0
    intercept = as_float(getattr(ds, "RescaleIntercept", 0.0), 0.0) or 0.0
    image = image * slope + intercept

    image = preprocess_numpy(image, clip_min=clip_min, clip_max=clip_max)
    image_u8 = np.rint(image * 255.0).astype(np.uint8)

    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        image_u8 = 255 - image_u8

    return image_u8


def save_png(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)


def convert_one_file(
    dcm_path: Path,
    input_root: Path,
    output_root: Path,
    overwrite: bool,
    clip_min: float,
    clip_max: float,
) -> int:
    import pydicom

    relative_path = dcm_path.relative_to(input_root)
    output_path = output_root / relative_path.with_suffix(".png")
    if output_path.exists() and not overwrite:
        return 0

    ds = pydicom.dcmread(str(dcm_path), force=True)
    pixel_array = ds.pixel_array

    if pixel_array.ndim == 2:
        save_png(
            dicom_to_preprocessed_uint8(ds, pixel_array, clip_min, clip_max),
            output_path,
        )
        return 1

    if pixel_array.ndim == 3:
        written = 0
        stem = output_path.stem
        for frame_idx, frame in enumerate(pixel_array):
            frame_path = output_path.with_name(f"{stem}_frame{frame_idx:03d}.png")
            if frame_path.exists() and not overwrite:
                continue
            save_png(
                dicom_to_preprocessed_uint8(ds, frame, clip_min, clip_max),
                frame_path,
            )
            written += 1
        return written

    raise ValueError(f"Unsupported pixel array shape: {pixel_array.shape}")


def iter_dicom_files(input_root: Path) -> list[Path]:
    return sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".dcm"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clip-min", type=float, default=DEFAULT_CLIP_MIN)
    parser.add_argument("--clip-max", type=float, default=DEFAULT_CLIP_MAX)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.exists():
        print(f"Input directory does not exist: {input_root}", file=sys.stderr)
        return 1

    try:
        import pydicom  # noqa: F401
    except ImportError:
        print("Missing dependency: pydicom. Install it with: pip install pydicom", file=sys.stderr)
        return 1

    dcm_files = iter_dicom_files(input_root)
    if args.max_files is not None:
        dcm_files = dcm_files[: args.max_files]

    print(f"Input root : {input_root}")
    print(f"Output root: {output_root}")
    print(f"DICOM files: {len(dcm_files)}")
    print(f"Clip range : [{args.clip_min}, {args.clip_max}]")

    converted = 0
    failed = 0
    for idx, dcm_path in enumerate(dcm_files, start=1):
        try:
            converted += convert_one_file(
                dcm_path=dcm_path,
                input_root=input_root,
                output_root=output_root,
                overwrite=args.overwrite,
                clip_min=args.clip_min,
                clip_max=args.clip_max,
            )
        except Exception as exc:
            failed += 1
            print(f"[WARN] Failed: {dcm_path} ({exc})", file=sys.stderr)

        if idx % 500 == 0 or idx == len(dcm_files):
            print(f"Processed {idx}/{len(dcm_files)} files, wrote {converted} PNGs")

    print(f"Done. Wrote {converted} PNGs. Failed DICOM files: {failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())