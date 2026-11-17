"""Inference over one split -> one prediction record per image."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..fdi import IDX2CODE


def predict_split(weights: str | Path, image_dir: str | Path, predict_args: dict) -> Iterator[dict]:
    """Yields {stem, size [w, h], boxes [[x1, y1, x2, y2]], codes [FDI], scores}.

    Boxes are in the pixel frame of the image on disk. `agnostic_nms` should be on in
    `predict_args`: one tooth must not come back twice under two different numbers.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    # A directory, not a list of paths: ultralytics turns a list into ONE batch of that length.
    for r in model.predict(str(image_dir), stream=True, verbose=False, **predict_args):
        h, w = r.orig_shape
        b = r.boxes
        yield dict(stem=Path(r.path).stem, size=[w, h],
                   boxes=[[round(v, 1) for v in x] for x in b.xyxy.tolist()],
                   codes=[IDX2CODE[int(c)] for c in b.cls.tolist()],
                   scores=[round(s, 4) for s in b.conf.tolist()])
