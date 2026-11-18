#!/usr/bin/env python
"""Step 5 -- table of the data-scaling test, next to the frozen-probe number of the same encoder.

    python scripts/05_scaling_report.py [--probe configs/probe/tooth_fdi_finetune.yaml] [--frozen configs/probe/tooth_fdi.yaml]
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from encprobe.config import all_encoder_cfgs, load_probe_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="configs/probe/tooth_fdi_finetune.yaml")
    ap.add_argument("--frozen", default="configs/probe/tooth_fdi.yaml")
    args = ap.parse_args()
    cfg, fz = load_probe_cfg(args.probe), load_probe_cfg(args.frozen)
    out = Path(cfg["out_root"]) / cfg["name"]
    frozen = {}
    fj = Path(fz["out_root"]) / fz["name"] / "results.json"
    if fj.exists():
        frozen = {r["encoder"]: r["test"]["fdi"] for r in json.loads(fj.read_text())["rows"]
                  if r["mode"] == "mlp" and r["size"] == cfg["image_size"]}
    fr = cfg["fractions"]
    for metric in ("fdi", "position", "quadrant"):
        print(f"\n## test {metric} accuracy, fine-tuned end to end at {cfg['image_size']} px\n")
        print("| encoder | frozen MLP (100%) | " + " | ".join(f"{f:.0%}" for f in fr) + " |")
        print("|---|---|" + "---|" * len(fr))
        for f in all_encoder_cfgs():
            cells = []
            for x in fr:
                p = out / "runs" / f"{f.stem}_f{x}.json"
                cells.append(str(json.loads(p.read_text())["test"][metric]) if p.exists() else "…")
            print(f"| {f.stem} | {frozen.get(f.stem, '–') if metric == 'fdi' else ''} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
