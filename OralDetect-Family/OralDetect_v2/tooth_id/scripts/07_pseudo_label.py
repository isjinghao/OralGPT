#!/usr/bin/env python
"""Step 7 -- number the teeth of image sets that have no tooth numbers (pseudo-labels from a trained run).

    python scripts/07_pseudo_label.py --data configs/data_unlabeled/intraoral_detection.yaml \
        --model configs/model/yolo11m_det.yaml [--only OMNI IO150K] [--n-viz 16] [--sheet-dir <folder> --tag v2]

Per dataset writes <run_dir>/pseudo_<config name>/<dataset>/{predictions.jsonl, instances_fdi.json,
viz_random/, viz_lowconf/} and, with --sheet-dir, two contact sheets there. A summary of all datasets
goes to <run_dir>/pseudo_<config name>/report.md. Nothing in the source datasets is touched.
"""
import argparse
import json
import os
import random
from pathlib import Path

import _bootstrap  # noqa: F401
from toothid.config import load_model_cfg, load_unlabeled_cfg, run_dir
from toothid.data.unlabeled import image_stats, list_images, to_coco
from toothid.engine.predict import predict_split
from toothid.viz.montage import montage
from toothid.viz.overlay import draw_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name")
    ap.add_argument("--weights", help="default: <run_dir>/weights/best.pt")
    ap.add_argument("--only", nargs="*", help="dataset names to run (default: all)")
    ap.add_argument("--n-viz", type=int, default=16)
    ap.add_argument("--sheet-dir", help="folder for the two contact sheets of each dataset")
    ap.add_argument("--tag", default="", help="suffix of the contact-sheet file names, e.g. v2")
    args = ap.parse_args()

    data, model = load_unlabeled_cfg(args.data), load_model_cfg(args.model)
    if args.name:
        model["name"] = args.name
    rd = run_dir(model)
    weights, root = args.weights or rd / "weights" / "best.pt", rd / f"pseudo_{data['name']}"
    rows, tag = [], f"_{args.tag}" if args.tag else ""
    for ds in data["datasets"]:
        if args.only and ds["name"] not in args.only:
            continue
        recs = {r["stem"]: r for r in list_images(ds)}
        out = root / ds["name"]
        stage = out / "images"                    # the predictor takes a directory: link the images into one
        stage.mkdir(parents=True, exist_ok=True)
        for r in recs.values():
            link = stage / Path(r["path"]).name
            if not link.exists():
                os.symlink(r["path"], link)

        preds = []
        with open(out / "predictions.jsonl", "w") as f:
            for p in predict_split(weights, stage, model["predict"]):
                p.update(image_stats(p))
                preds.append(p)
                f.write(json.dumps(p) + "\n")
        (out / "instances_fdi.json").write_text(json.dumps(to_coco(preds, {s: Path(r["path"]).name for s, r in recs.items()})))

        n = len(preds)
        some = [p for p in preds if p["n"]]
        rows.append(dict(dataset=ds["name"], images=n, no_tooth=n - len(some), teeth=sum(p["n"] for p in preds),
                         teeth_per_image=round(sum(p["n"] for p in some) / max(len(some), 1), 1),
                         mean_score=round(sum(p["mean_score"] for p in some) / max(len(some), 1), 3),
                         repeat_images=sum(p["n_repeat"] > 0 for p in preds)))
        print(json.dumps(rows[-1]), flush=True)

        picks = dict(random=random.Random(0).sample(preds, min(args.n_viz, n)),
                     lowconf=sorted(some, key=lambda p: p["mean_score"])[: args.n_viz])
        for kind, sample in picks.items():
            folder = out / f"viz_{kind}"
            folder.mkdir(exist_ok=True)
            files = []
            for p in sample:
                files.append(folder / f"{p['stem']}.jpg")
                draw_pred(recs[p["stem"]]["path"], p["codes"], p["boxes"], p["scores"]).save(files[-1], quality=90)
            if args.sheet_dir and files:
                what = "random" if kind == "random" else "lowest mean confidence"
                montage(files, Path(args.sheet_dir) / f"{ds['name']}__pseudo_{kind}{len(files)}{tag}.jpg", strip=ds.get("prefix", ""),
                        title=f"{ds['name']} / {model['name']}: predicted FDI numbers, {len(files)} {what} images of {n} "
                              f"(yellow Q1, cyan Q2, green Q3, pink Q4; second line = confidence)")

    keys = list(rows[0])
    table = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)] + ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    note = ("`no_tooth` = images where nothing was detected; `teeth_per_image` and `mean_score` are over the other images; "
            "`repeat_images` = images where one FDI number was given to two or more boxes.")
    (root / "report.md").write_text(f"# Pseudo FDI labels: {model['name']} on {data['name']}\n\nweights: {weights}\n\n" + "\n".join(table) + f"\n\n{note}\n")
    print("report:", root / "report.md")


if __name__ == "__main__":
    main()
