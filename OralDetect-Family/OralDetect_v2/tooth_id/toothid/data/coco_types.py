"""Reader for COCO-format sets with human tooth boxes but no full FDI number, as packaged for OralDetect v1."""
from __future__ import annotations

import json
from pathlib import Path

from ..fdi import NAME2POSITION


def load_records(cfg: dict) -> list[dict]:
    """[{stem, path, split, positions, boxes [[x1, y1, x2, y2]]}], one per image with at least one kept box.

    Without `categories` in the config, the kept categories are the tooth types and `positions` holds
    their position digit (1..8). With `categories: [names]`, those are kept and `positions` holds 0.
    Boxes of any other category are ignored."""
    value = {n: 0 for n in cfg["categories"]} if cfg.get("categories") else NAME2POSITION
    out = []
    for src in cfg["sources"]:
        d = json.load(open(src["annotations"]))
        pos = {c["id"]: value[c["name"]] for c in d["categories"] if c["name"] in value}
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
