#!/usr/bin/env python
"""Step 4 -- draw predictions from step 3 against ground truth.

    python scripts/04_visualize.py --data ... --model ... [--split test] [--n 20] [--worst]

--worst picks the images with the most wrong numbers instead of a random sample.
Writes <run_dir>/eval_<split>/viz/<stem>.jpg.
"""
import argparse
import json
import random
from pathlib import Path

import _bootstrap  # noqa: F401
from toothid.config import load_data_cfg, load_model_cfg, run_dir
from toothid.data.yolo_format import read_label
from toothid.eval.matching import greedy_match
from toothid.viz.overlay import draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--name")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--worst", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data, model = load_data_cfg(args.data), load_model_cfg(args.model)
    if args.name:
        model["name"] = args.name
    root, out = Path(data["yolo_root"]), run_dir(model) / f"eval_{args.split}"
    thr = model["eval"]["match_iou"]
    preds = [json.loads(l) for l in open(out / "predictions.jsonl")]

    def gt(p):
        return read_label(root / "labels" / args.split / f"{p['stem']}.txt", *p["size"])

    if args.worst:
        def n_wrong(p):
            codes, boxes = gt(p)
            pairs = greedy_match(boxes, p["boxes"], thr) if boxes and p["boxes"] else []
            return sum(codes[g] != p["codes"][q] for g, q, _ in pairs)
        preds.sort(key=n_wrong, reverse=True)
    else:
        random.Random(args.seed).shuffle(preds)

    (out / "viz").mkdir(exist_ok=True)
    for p in preds[: args.n]:
        codes, boxes = gt(p)
        im = draw(root / "images" / args.split / f"{p['stem']}.jpg", codes, boxes, p["codes"], p["boxes"], thr)
        im.save(out / "viz" / f"{p['stem']}.jpg", quality=90)
    print("wrote", min(args.n, len(preds)), "images to", out / "viz")


if __name__ == "__main__":
    main()
