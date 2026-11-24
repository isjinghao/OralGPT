#!/usr/bin/env python
"""Step 8 -- training set = an existing FDI YOLO dataset + sets with human boxes but incomplete numbers.

    python scripts/08_build_typed_union.py --data configs/data/iop_plus_io150k.yaml

Writes <yolo_root>/{images, labels, dataset.yaml, union_report.json}. The base dataset is linked,
not copied; each extra set's images are linked and its labels written (see toothid/data/typed_union.py).
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image

import _bootstrap  # noqa: F401
from toothid.config import load_external_cfg, load_union_cfg
from toothid.data.coco_types import load_records
from toothid.data.typed_union import anterior12_codes, box_label_line, complete_codes, other_teeth
from toothid.data.yolo_format import write_dataset_yaml
from toothid.fdi import CODE2IDX


def link(src: Path, dst: Path):
    if not dst.is_symlink() and not dst.exists():
        os.symlink(src, dst)


def number(ex: dict, r: dict, p: dict | None):
    """-> (codes, boxes, n_borrowed) for one image, or a string saying why it is dropped."""
    if p is None:
        return "no prediction"
    if ex["kind"] == "tooth_type":
        codes = complete_codes(r["positions"], r["boxes"], p["codes"], p["boxes"], ex["match_iou"])
        return (codes, r["boxes"], 0) if codes else "quadrant unclear or number repeated"
    if ex["kind"] == "anterior12":
        codes = anterior12_codes(r["boxes"])
        if codes is None:
            return "not 12 boxes"
        more = other_teeth(r["boxes"], codes, p["codes"], p["boxes"], p["scores"], ex["match_iou"], ex["min_score"])
        return (codes + more[0], r["boxes"] + more[1], len(more[0])) if more is not None else "model disagrees with layout number"
    raise ValueError(f"unknown kind {ex['kind']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    cfg = load_union_cfg(args.data)
    root, base = Path(cfg["yolo_root"]), Path(cfg["base_yolo_root"])
    for kind in ("images", "labels"):
        (root / kind).mkdir(parents=True, exist_ok=True)
        link(base / kind / "train", root / kind / "train_base")
        for s in ("val", "test"):
            link(base / kind / s, root / kind / s)

    train, report = ["images/train_base"], dict(base_train_images=len(list((base / "images" / "train").iterdir())), extras=[])
    for ex in cfg["extras"]:
        folder = f"train_{ex['name']}"
        for kind in ("images", "labels"):
            (root / kind / folder).mkdir(exist_ok=True)
        preds = {p["stem"]: p for p in map(json.loads, open(ex["preds_from"]))}
        recs = [r for r in load_records(load_external_cfg(ex["config"])) if r["split"] in ex["train_splits"]]
        kept, borrowed, quad, why = 0, 0, Counter(), Counter()
        for r in recs:
            res = number(ex, r, preds.get(r["stem"]))
            if isinstance(res, str):
                why[res] += 1
                continue
            codes, boxes, nb = res
            w, h = Image.open(r["path"]).size
            link(Path(r["path"]), root / "images" / folder / Path(r["path"]).name)
            (root / "labels" / folder / f"{r['stem']}.txt").write_text(
                "\n".join(box_label_line(CODE2IDX[c], b, w, h) for c, b in zip(codes, boxes)) + "\n")
            kept, borrowed = kept + 1, borrowed + nb
            quad.update(c // 10 for c in codes)
        train.append(f"images/{folder}")
        report["extras"].append(dict(name=ex["name"], kind=ex["kind"], candidates=len(recs), kept=kept, not_kept=dict(why),
                                     teeth=sum(quad.values()), teeth_from_model=borrowed,
                                     teeth_per_quadrant={str(q): quad[q] for q in sorted(quad)}))

    yaml_path = write_dataset_yaml(root, {"train": train, "val": "images/val", "test": "images/test"})
    (root / "union_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("dataset yaml:", yaml_path)


if __name__ == "__main__":
    main()
