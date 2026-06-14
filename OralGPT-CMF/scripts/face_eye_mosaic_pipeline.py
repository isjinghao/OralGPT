#!/usr/bin/env python3
"""
Face eye mosaic pipeline for SH9HCMFdata.

Pipeline:
1) Recursively scan all `FP` folders under dataset root.
2) Detect face landmarks with MediaPipe Face Mesh.
3) Build polygon mask from eye landmarks (left + right eyes).
4) Apply mosaic or gaussian blur inside eye regions.
5) Save to sibling output folder (default: FP_clean) with same filenames.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


# Eye contour indices in MediaPipe Face Mesh (468 landmarks model)
# These are stable contour points around left/right eyes.
LEFT_EYE_IDX = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]

RIGHT_EYE_IDX = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def get_face_mesh_class():
    """
    Resolve MediaPipe FaceMesh class across different package layouts.
    """
    try:
        import mediapipe as mp  # type: ignore

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            return mp.solutions.face_mesh.FaceMesh
    except Exception:
        pass

    # Fallback path used by some installations.
    try:
        mod = importlib.import_module("mediapipe.python.solutions.face_mesh")
        return mod.FaceMesh
    except Exception as exc:
        raise RuntimeError(
            "Cannot import MediaPipe FaceMesh. "
            "Please install a compatible mediapipe package, e.g.:\n"
            "  pip install --upgrade mediapipe\n"
            "or if needed:\n"
            "  pip install mediapipe==0.10.14"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eye-only mosaic pipeline using MediaPipe Face Mesh + OpenCV."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/OralGPT/OralGPT-CMF/dataset/SH9HCMFdata"),
        help="Dataset root path that contains group folders.",
    )
    parser.add_argument(
        "--input-folder-name",
        type=str,
        default="FP",
        help="Input image folder name under each user folder.",
    )
    parser.add_argument(
        "--output-folder-name",
        type=str,
        default="FP_mosaic",
        help="Output folder name (sibling of FP).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mosaic", "blur"],
        default="mosaic",
        help="Region anonymization mode: mosaic or blur.",
    )
    parser.add_argument(
        "--mosaic-scale",
        type=float,
        default=0.08,
        help="Downscale ratio for mosaic (smaller -> stronger mosaic).",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=31,
        help="Gaussian blur kernel size (odd positive int) when --mode blur.",
    )
    parser.add_argument(
        "--eye-padding",
        type=float,
        default=0.18,
        help="Eye polygon expansion ratio around eye center.",
    )
    parser.add_argument(
        "--max-num-faces",
        type=int,
        default=1,
        help="Max faces to process per image.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.35,
        help="Face Mesh detection confidence threshold (lower -> higher recall).",
    )
    parser.add_argument(
        "--detect-short-side",
        type=int,
        default=960,
        help="Upscale image to this short-side size before landmark detection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser.parse_args()


def gather_fp_dirs(root: Path, input_folder_name: str) -> List[Path]:
    return sorted(
        p for p in root.rglob(input_folder_name) if p.is_dir() and p.name == input_folder_name
    )


def gather_images(folder: Path) -> List[Path]:
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def landmarks_to_points(
    landmarks, image_w: int, image_h: int, indices: Sequence[int]
) -> np.ndarray:
    points = []
    for idx in indices:
        lm = landmarks[idx]
        x = int(np.clip(lm.x * image_w, 0, image_w - 1))
        y = int(np.clip(lm.y * image_h, 0, image_h - 1))
        points.append([x, y])
    return np.array(points, dtype=np.int32)


def expand_polygon(points: np.ndarray, ratio: float) -> np.ndarray:
    if len(points) == 0:
        return points
    center = points.mean(axis=0, keepdims=True)
    expanded = center + (points - center) * (1.0 + ratio)
    return np.round(expanded).astype(np.int32)


def apply_mosaic_to_mask(image: np.ndarray, mask: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    scale = float(np.clip(scale, 0.02, 1.0))
    small_w = max(1, int(w * scale))
    small_h = max(1, int(h * scale))
    down = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)
    out = image.copy()
    out[mask > 0] = mosaic[mask > 0]
    return out


def apply_blur_to_mask(image: np.ndarray, mask: np.ndarray, kernel: int) -> np.ndarray:
    if kernel < 1:
        kernel = 1
    if kernel % 2 == 0:
        kernel += 1
    blurred = cv2.GaussianBlur(image, (kernel, kernel), sigmaX=0)
    out = image.copy()
    out[mask > 0] = blurred[mask > 0]
    return out


def is_valid_eye_polygon(points: np.ndarray, image_w: int, image_h: int) -> bool:
    if points.shape[0] < 3:
        return False
    area = float(cv2.contourArea(points))
    min_area = max(40.0, image_w * image_h * 0.00002)
    return area >= min_area


def maybe_upscale_for_detection(
    image_bgr: np.ndarray, target_short_side: int
) -> Tuple[np.ndarray, float]:
    h, w = image_bgr.shape[:2]
    short_side = min(h, w)
    if target_short_side <= 0 or short_side >= target_short_side:
        return image_bgr, 1.0

    scale = float(target_short_side) / float(short_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    upscaled = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return upscaled, scale


def anonymize_eyes(
    image_bgr: np.ndarray,
    face_mesh,
    mode: str,
    mosaic_scale: float,
    blur_kernel: int,
    eye_padding: float,
    detect_short_side: int,
) -> Tuple[np.ndarray, bool]:
    h, w = image_bgr.shape[:2]
    detect_img, detect_scale = maybe_upscale_for_detection(image_bgr, detect_short_side)
    dh, dw = detect_img.shape[:2]

    rgb = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return image_bgr, False

    detect_mask = np.zeros((dh, dw), dtype=np.uint8)
    any_eye_valid = False
    for face_landmarks in results.multi_face_landmarks:
        lm = face_landmarks.landmark
        left_eye = landmarks_to_points(lm, dw, dh, LEFT_EYE_IDX)
        right_eye = landmarks_to_points(lm, dw, dh, RIGHT_EYE_IDX)
        left_eye = expand_polygon(left_eye, eye_padding)
        right_eye = expand_polygon(right_eye, eye_padding)

        # Allow one-eye-only anonymization for profile faces.
        if is_valid_eye_polygon(left_eye, dw, dh):
            cv2.fillPoly(detect_mask, [left_eye], 255)
            any_eye_valid = True
        if is_valid_eye_polygon(right_eye, dw, dh):
            cv2.fillPoly(detect_mask, [right_eye], 255)
            any_eye_valid = True

    if not any_eye_valid:
        return image_bgr, False

    if detect_scale != 1.0:
        mask = cv2.resize(detect_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        mask = detect_mask

    if mode == "mosaic":
        return apply_mosaic_to_mask(image_bgr, mask, mosaic_scale), True
    return apply_blur_to_mask(image_bgr, mask, blur_kernel), True


def process_dataset(args: argparse.Namespace) -> int:
    if not args.root.exists():
        print(f"[ERROR] root not found: {args.root}")
        return 1

    fp_dirs = gather_fp_dirs(args.root, args.input_folder_name)
    if not fp_dirs:
        print(
            f"[WARN] no '{args.input_folder_name}' folders found under: {args.root}"
        )
        return 0

    total_images = 0
    total_processed = 0
    total_no_face = 0
    total_failed = 0

    FaceMesh = get_face_mesh_class()
    with FaceMesh(
        static_image_mode=True,
        max_num_faces=args.max_num_faces,
        refine_landmarks=True,
        min_detection_confidence=args.min_detection_confidence,
    ) as face_mesh:
        for fp_dir in fp_dirs:
            out_dir = fp_dir.parent / args.output_folder_name
            out_dir.mkdir(parents=True, exist_ok=True)

            images = gather_images(fp_dir)
            if not images:
                continue

            print(f"[INFO] Processing folder: {fp_dir}")
            for img_path in images:
                total_images += 1
                out_path = out_dir / img_path.name
                if out_path.exists() and not args.overwrite:
                    continue

                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"  [FAIL] cannot read: {img_path}")
                    total_failed += 1
                    continue

                out_img, ok = anonymize_eyes(
                    image_bgr=image,
                    face_mesh=face_mesh,
                    mode=args.mode,
                    mosaic_scale=args.mosaic_scale,
                    blur_kernel=args.blur_kernel,
                    eye_padding=args.eye_padding,
                    detect_short_side=args.detect_short_side,
                )

                if not ok:
                    # Preserve original image if no face detected.
                    out_img = image
                    total_no_face += 1

                success = cv2.imwrite(str(out_path), out_img)
                if not success:
                    print(f"  [FAIL] cannot write: {out_path}")
                    total_failed += 1
                    continue

                total_processed += 1

    print("\n[SUMMARY]")
    print(f"  FP folders found   : {len(fp_dirs)}")
    print(f"  images seen        : {total_images}")
    print(f"  images written     : {total_processed}")
    print(f"  no face detected   : {total_no_face}")
    print(f"  read/write failures: {total_failed}")
    return 0


def main() -> None:
    args = parse_args()
    code = process_dataset(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
