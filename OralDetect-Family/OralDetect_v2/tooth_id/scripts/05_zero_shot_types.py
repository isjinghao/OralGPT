#!/usr/bin/env python
"""Step 5 -- zero-shot test of a trained run on an external set labelled with tooth types.

    python scripts/05_zero_shot_types.py --data configs/data_external/io150k.yaml \
        --model configs/model/yolo11m_det.yaml [--name ...] [--weights ...] [--n-viz 16]

Writes <run_dir>/zeroshot_<data name>/{predictions.jsonl, metrics.json, report.md, viz_worst/, viz_random/}.
"""
import argparse
import json
import os
import random
from pathlib import Path

import _bootstrap  # noqa: F401
from toothid.config import load_external_cfg, load_model_cfg, run_dir
from toothid.data.coco_types import load_records
from toothid.engine.predict import predict_split
from toothid.eval.matching import greedy_match
from toothid.eval.type_metrics import TypeAccumulator, to_markdown
from toothid.fdi import position, quadrant
from toothid.viz.overlay import draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name")
    ap.add_argument("--weights", help="default: <run_dir>/weights/best.pt")
    ap.add_argument("--n-viz", type=int, default=16)
    args = ap.parse_args()

    data, model = load_external_cfg(args.data), load_model_cfg(args.model)
    if args.name:
        model["name"] = args.name
    rd = run_dir(model)
    weights, out, thr = args.weights or rd / "weights" / "best.pt", rd / f"zeroshot_{data['name']}", model["eval"]["match_iou"]
    recs = {r["stem"]: r for r in load_records(data)}

    stage = out / "images"                    # the predictor takes a directory: link the selected images into one
    stage.mkdir(parents=True, exist_ok=True)
    for r in recs.values():
        link = stage / Path(r["path"]).name
        if not link.exists():
            os.symlink(r["path"], link)

    acc, preds = TypeAccumulator(thr), []
    with open(out / "predictions.jsonl", "w") as f:
        for p in predict_split(weights, stage, model["predict"]):
            r = recs[p["stem"]]
            pairs = acc.add(r["split"], r["positions"], r["boxes"], p["codes"], p["boxes"])
            wrong = sum(r["positions"][g] != position(p["codes"][q]) for g, q, _ in pairs)
            p["n_err"] = wrong + (len(r["boxes"]) - len(pairs)) + (len(p["boxes"]) - len(pairs))
            preds.append(p)
            f.write(json.dumps(p) + "\n")

    metrics = dict(run=model["name"], data=data["name"], n_images=len(preds), weights=str(weights), tooth_type=acc.summary())
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "report.md").write_text(to_markdown(metrics["tooth_type"], f"{model['name']} / zero-shot on {data['name']} ({len(preds)} images)"))

    def viz(sample, folder):
        (out / folder).mkdir(exist_ok=True)
        for p in sample:
            r = recs[p["stem"]]
            # The overlay compares full codes. Ground truth has no quadrant, so borrow the matched
            # prediction's: a box then shows green exactly when the position digit is right.
            gt = list(r["positions"])
            for g, q, _ in greedy_match(r["boxes"], p["boxes"], thr) if p["boxes"] else []:
                gt[g] = quadrant(p["codes"][q]) * 10 + r["positions"][g]
            draw(r["path"], gt, r["boxes"], p["codes"], p["boxes"], thr, p["scores"]).save(out / folder / f"{p['stem']}.jpg", quality=90)

    viz(sorted(preds, key=lambda p: -p["n_err"])[: args.n_viz], "viz_worst")
    viz(random.Random(0).sample(preds, min(args.n_viz, len(preds))), "viz_random")
    print(json.dumps(metrics["tooth_type"]["overall"], indent=1))
    print("report:", out / "report.md")


if __name__ == "__main__":
    main()
