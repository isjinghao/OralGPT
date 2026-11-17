#!/usr/bin/env python
"""Step 3 -- evaluate a run on one split: COCO-style mAP (ultralytics) + tooth-ID metrics (ours).

    python scripts/03_evaluate.py --data configs/data/iop_compass.yaml \
        --model configs/model/yolo11m_seg.yaml [--split test] [--name smoke]

Writes <run_dir>/eval_<split>/{metrics.json, report.md, predictions.jsonl}.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from toothid.config import load_data_cfg, load_model_cfg, run_dir
from toothid.data.yolo_format import read_label, view_of
from toothid.engine.predict import predict_split
from toothid.eval.id_metrics import IDAccumulator, to_markdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--name")
    ap.add_argument("--weights", help="default: <run_dir>/weights/best.pt")
    args = ap.parse_args()

    data, model = load_data_cfg(args.data), load_model_cfg(args.model)
    if args.name:
        model["name"] = args.name
    rd, root = run_dir(model), Path(data["yolo_root"])
    weights = args.weights or rd / "weights" / "best.pt"
    out = rd / f"eval_{args.split}"
    out.mkdir(parents=True, exist_ok=True)

    # (a) mAP, per-class-aware, straight from ultralytics
    from ultralytics import YOLO
    v = YOLO(str(weights)).val(data=str(root / "dataset.yaml"), split=args.split, imgsz=model["predict"]["imgsz"],
                               project=str(out), name="ultralytics_val", exist_ok=True, plots=True, verbose=False)
    coco = dict(box_map50_95=round(float(v.box.map), 4), box_map50=round(float(v.box.map50), 4),
                mask_map50_95=round(float(v.seg.map), 4), mask_map50=round(float(v.seg.map50), 4))

    # (b) tooth-ID metrics on class-agnostic matches
    acc = IDAccumulator(model["eval"]["match_iou"])
    with open(out / "predictions.jsonl", "w") as f:
        for p in predict_split(weights, root / "images" / args.split, model["predict"]):
            gt_codes, gt_boxes = read_label(root / "labels" / args.split / f"{p['stem']}.txt", *p["size"])
            acc.add(view_of(p["stem"]), gt_codes, gt_boxes, p["codes"], p["boxes"])
            f.write(json.dumps(p) + "\n")

    metrics = dict(run=model["name"], split=args.split, weights=str(weights), coco=coco, tooth_id=acc.summary())
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    md = to_markdown(metrics["tooth_id"], f"{model['name']} / {args.split}")
    md += "\n## mAP (ultralytics)\n\n| metric | value |\n|---|---|\n" + "".join(f"| {k} | {x} |\n" for k, x in coco.items())
    (out / "report.md").write_text(md)
    print(json.dumps(dict(coco=coco, overall=metrics["tooth_id"]["overall"]), indent=1))
    print("report:", out / "report.md")


if __name__ == "__main__":
    main()
