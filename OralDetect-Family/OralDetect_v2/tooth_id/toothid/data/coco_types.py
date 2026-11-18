"""Reader for COCO-format sets whose categories are tooth types (no quadrant), e.g. IO150K."""
from __future__ import annotations

import json
from pathlib import Path

from ..fdi import NAME2POSITION


def load_records(cfg: dict) -> list[dict]:
    """[{stem, path, split, positions [1..8], boxes [[x1, y1, x2, y2]]}], one per image that has at
    least one tooth-type box. Boxes of any other category are ignored."""
    out = []
    for src in cfg["sources"]:
        d = json.load(open(src["annotations"]))
        pos = {c["id"]: NAME2POSITION[c["name"]] for c in d["categories"] if c["name"] in NAME2POSITION}
        imgs = {im["id"]: Path(im["file_name"]).name for im in d["images"]
                if Path(im["file_name"]).name.startswith(cfg.get("prefix", ""))}
        recs = {i: dict(stem=Path(n).stem, path=str(Path(src["images"]) / n), split=src["split"],
                        positions=[], boxes=[]) for i, n in imgs.items()}
        for a in d["annotations"]:
            if a["image_id"] in recs and a["category_id"] in pos:
                x, y, w, h = a["bbox"]
                recs[a["image_id"]]["positions"].append(pos[a["category_id"]])
                recs[a["image_id"]]["boxes"].append([x, y, x + w, y + h])
        out += [r for r in recs.values() if r["boxes"]]
    return out
