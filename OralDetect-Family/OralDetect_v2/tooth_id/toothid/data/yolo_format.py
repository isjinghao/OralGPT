"""Ultralytics YOLO-seg dataset on disk: writer and reader.

<root>/images/<split>/<stem>.jpg     resized so the long side is <= max_side
<root>/labels/<split>/<stem>.txt     one line per tooth: <class_idx> x1 y1 x2 y2 ... (normalised)
<root>/dataset.yaml                  what ultralytics reads
<root>/manifest.jsonl                one record per image (patient, view, split, sizes, n_teeth)
"""
from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from ..fdi import CLASS_NAMES, CODE2IDX, IDX2CODE
from .iop_compass import Record, load_teeth

Image.MAX_IMAGE_PIXELS = None


def write_sample(rec: Record, split: str, root: str | Path, max_side: int, jpeg_quality: int) -> dict:
    root = Path(root)
    im = Image.open(rec.image).convert("RGB")
    w0, h0 = im.size
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    img_out = root / "images" / split / f"{rec.stem}.jpg"
    im.save(img_out, quality=jpeg_quality)

    lines = []
    for t in load_teeth(rec.ann):
        xy = " ".join(f"{min(max(x / w0, 0), 1):.6f} {min(max(y / h0, 0), 1):.6f}" for x, y in t.polygon)
        lines.append(f"{CODE2IDX[t.code]} {xy}")
    (root / "labels" / split / f"{rec.stem}.txt").write_text("\n".join(lines) + "\n")
    return dict(stem=rec.stem, patient=rec.patient, view=rec.view, split=split,
                orig_size=[w0, h0], size=list(im.size), n_teeth=len(lines))


def write_dataset_yaml(root: str | Path, splits: list[str] | dict) -> Path:
    """`splits`: split names (each at images/<split>), or {split: folder or list of folders}."""
    root = Path(root)
    if not isinstance(splits, dict):
        splits = {s: f"images/{s}" for s in splits}
    # No `path:` key -- ultralytics then resolves the splits against this file's own directory, so
    # the dataset can be moved or mounted elsewhere without a rewrite.
    cfg = dict(**splits, names=dict(enumerate(CLASS_NAMES)))
    out = root / "dataset.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def read_label(path: str | Path, w: int, h: int) -> tuple[list[int], list[list[float]]]:
    """-> (FDI codes, xyxy boxes in pixels) -- the box is the polygon's extent."""
    codes, boxes = [], []
    for line in Path(path).read_text().splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        xs, ys = [float(a) * w for a in v[1::2]], [float(a) * h for a in v[2::2]]
        codes.append(IDX2CODE[int(v[0])])
        boxes.append([min(xs), min(ys), max(xs), max(ys)])
    return codes, boxes


def view_of(stem: str) -> str:
    return stem.split("_")[1]
