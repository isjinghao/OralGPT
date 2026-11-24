#!/usr/bin/env python
"""Step 7 -- the bench retrieval of step 6, for the encoders fine-tuned in step 4 (every fraction).

    python scripts/07_bench_retrieval_finetuned.py [--only vit_b_dinov3] [--probe ...] [--finetune ...]

Incremental: scores every checkpoint that exists and is not scored yet, one result file per encoder
(<out_root>/<probe>/finetuned/<encoder>.json), so jobs for different encoders can run side by side.
Rewrites <out_root>/<probe>/finetuned_report.md from whatever is there; the pretrained column comes from step 6.
"""
import argparse
import json
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from encprobe.config import all_encoder_cfgs, load_encoder_cfg, load_probe_cfg
from encprobe.encoders import build_finetuned
from encprobe.retrieval import BenchImages, global_features, retrieval_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--probe", default="configs/probe/bench_retrieval.yaml")
    ap.add_argument("--finetune", default="configs/probe/tooth_fdi_finetune.yaml")
    args = ap.parse_args()
    cfg, ft = load_probe_cfg(args.probe), load_probe_cfg(args.finetune)
    out, runs = Path(cfg["out_root"]) / cfg["name"], Path(ft["out_root"]) / ft["name"] / "runs"
    (out / "finetuned").mkdir(parents=True, exist_ok=True)
    for f in all_encoder_cfgs():
        if args.only and f.stem not in args.only:
            continue
        enc, done = load_encoder_cfg(f), out / "finetuned" / f"{f.stem}.json"
        rows = json.loads(done.read_text()) if done.exists() else []
        have = {(r["fraction"], r["size"]) for r in rows}
        for size in cfg["image_sizes"]:
            ds = BenchImages(cfg["bench_root"], size)
            for frac in ft["fractions"]:
                ckpt, meta = runs / f"{f.stem}_f{frac}.pt", runs / f"{f.stem}_f{frac}.json"
                if not (ckpt.exists() and meta.exists()) or (frac, size) in have:      # .json is written when the run is over
                    continue
                try:
                    model = build_finetuned(enc, ckpt, ft["image_size"], size).cuda()
                except ValueError as e:
                    print(f"skip {f.stem} @ {size}: {e}")
                    break
                s = retrieval_scores(global_features(model, ds, cfg["batch"], cfg["workers"], cfg["amp"]), ds.rows)
                rows.append(dict(encoder=f.stem, fraction=frac, size=size, scores=s))
                print(f.stem, frac, size, s["macro"], flush=True)
                done.write_text(json.dumps(rows, indent=1))
                del model
                torch.cuda.empty_cache()

    base = {(r["encoder"], r["size"]): r["scores"] for r in json.loads((out / "results.json").read_text())}
    got = {(r["encoder"], r["fraction"], r["size"]): r["scores"] for p in sorted((out / "finetuned").glob("*.json"))
           for r in json.loads(p.read_text())}
    md = ["# Bench image retrieval after fine-tuning on tooth numbering", "",
          "Encoders fine-tuned end to end on a fraction of IOP-Compass (FDI classification of tooth boxes, 1024 px), then scored "
          "on OralDetect-Bench image-to-image retrieval exactly as the pretrained weights were. Macro mean over 5 modalities. "
          "`…` = not run yet.", ""]
    for size in cfg["image_sizes"]:
        for metric in ("R@1", "mAP"):
            md += [f"## {metric}, input {size}", "", "| encoder | pretrained | " + " | ".join(f"{x:.0%}" for x in ft["fractions"]) + " |",
                   "|---|---|" + "---|" * len(ft["fractions"])]
            for f in all_encoder_cfgs():
                if (f.stem, size) not in base:
                    continue
                cells = [str(got[(f.stem, x, size)]["macro"][metric]) if (f.stem, x, size) in got else "…" for x in ft["fractions"]]
                md.append(f"| {f.stem} | {base[(f.stem, size)]['macro'][metric]} | " + " | ".join(cells) + " |")
            md.append("")
    (out / "finetuned_report.md").write_text("\n".join(md))
    print("report:", out / "finetuned_report.md")


if __name__ == "__main__":
    main()
