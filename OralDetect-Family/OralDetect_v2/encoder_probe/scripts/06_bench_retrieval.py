#!/usr/bin/env python
"""Step 6 -- image-to-image retrieval on OralDetect-Bench for every encoder.

    python scripts/06_bench_retrieval.py [--probe configs/probe/bench_retrieval.yaml]

Writes <out_root>/<probe>/{results.json, report.md}.
"""
import argparse
import json
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from encprobe.config import all_encoder_cfgs, load_encoder_cfg, load_probe_cfg
from encprobe.encoders import build
from encprobe.retrieval import BenchImages, global_features, retrieval_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="configs/probe/bench_retrieval.yaml")
    args = ap.parse_args()
    cfg = load_probe_cfg(args.probe)
    out = Path(cfg["out_root"]) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for size in cfg["image_sizes"]:
        ds = BenchImages(cfg["bench_root"], size)
        for f in all_encoder_cfgs():
            enc = load_encoder_cfg(f)
            try:
                model = build(enc, size).cuda()
            except ValueError as e:
                print(f"skip {enc['name']} @ {size}: {e}")
                continue
            s = retrieval_scores(global_features(model, ds, cfg["batch"], cfg["workers"], cfg["amp"]), ds.rows)
            rows_out.append(dict(encoder=enc["name"], backbone=enc["backbone"], pretrain=enc["pretrain"], size=size, scores=s))
            print(enc["name"], size, s["macro"], flush=True)
            del model
            torch.cuda.empty_cache()
    (out / "results.json").write_text(json.dumps(rows_out, indent=1))

    mods = [m for m in rows_out[0]["scores"] if m != "macro"]
    md = ["# Image-to-image retrieval on OralDetect-Bench (Retrieval side)", "",
          "Frozen encoders, no training. Feature = mean-pooled last feature map, centred and L2-normalised. Each image queries "
          "the other images of its modality; a hit shares its `category`. `macro` = mean over the modalities.", "",
          "| modality | queries | categories | largest category | chance R@1 |", "|---|---|---|---|---|"]
    s0 = rows_out[0]["scores"]
    md += [f"| {m} | {s0[m]['n_queries']} | {s0[m]['n_categories']} | {s0[m]['largest_category']} | {s0[m]['chance']} |" for m in mods]
    for size in cfg["image_sizes"]:
        for metric in ("R@1", "mAP"):
            md += ["", f"## {metric}, input {size}", "", "| encoder | macro | " + " | ".join(mods) + " |", "|---|---|" + "---|" * len(mods)]
            md += [f"| {r['encoder']} | {r['scores']['macro'][metric]} | " + " | ".join(str(r["scores"][m][metric]) for m in mods) + " |"
                   for r in rows_out if r["size"] == size]
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("report:", out / "report.md")


if __name__ == "__main__":
    main()
