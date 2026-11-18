#!/usr/bin/env python
"""Step 3 -- train the probes on every cached feature file and write the comparison table.

    python scripts/03_probe.py --probe configs/probe/tooth_fdi.yaml

Writes <out_root>/<probe>/{results.json, report.md}.
"""
import argparse
import json
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from encprobe.config import all_encoder_cfgs, feature_file, load_encoder_cfg, load_probe_cfg
from encprobe.probe import run_probe

COLS = ("fdi", "position", "quadrant", "arch", "side")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    args = ap.parse_args()
    probe = load_probe_cfg(args.probe)
    out, dev = Path(probe["out_root"]) / probe["name"], "cuda" if torch.cuda.is_available() else "cpu"
    rows, floor = [], None
    for size in probe["image_sizes"]:
        for f in all_encoder_cfgs():
            enc = load_encoder_cfg(f)
            ff = feature_file(probe, enc["name"], size)
            if not ff.exists():
                continue
            d = torch.load(ff, map_location="cpu", weights_only=False)
            if floor is None:                         # geometry does not depend on the encoder or the size
                floor = run_probe(d["train"], d["val"], d["test"], "geom", probe["probe"], dev)
                print("geom floor", floor["test"], flush=True)
            for mode in ("linear", "mlp", "mlp+geom"):
                r = run_probe(d["train"], d["val"], d["test"], mode, probe["probe"], dev)
                rows.append(dict(encoder=enc["name"], backbone=enc["backbone"], pretrain=enc["pretrain"], size=size,
                                 dim=int(d["train"]["feat"].shape[1]), **r))
                print(enc["name"], size, mode, r["test"]["fdi"], flush=True)
    (out / "results.json").write_text(json.dumps(dict(floor=floor, rows=rows), indent=1))

    md = [f"# Frozen-encoder probe: {probe['name']}", "",
          "Each ground-truth tooth box of IOP-Compass is RoI-pooled from the frozen encoder's last feature map and "
          "classified into its FDI code (32 classes). Test split, model selected on val. "
          "`position` = which tooth in the quadrant, `quadrant` = which of the four, `arch` = upper/lower, `side` = left/right.", "",
          f"Floor (box position, size and view only, no image feature): " + ", ".join(f"{c} {floor['test'][c]}" for c in COLS), ""]
    for size in probe["image_sizes"]:
        for mode in ("linear", "mlp", "mlp+geom"):
            sel = [r for r in rows if r["size"] == size and r["mode"] == mode]
            if not sel:
                continue
            md += [f"## {mode}, input {size}", "", "| encoder | pretraining | dim | " + " | ".join(COLS) + " |", "|---|---|---|" + "---|" * len(COLS)]
            md += [f"| {r['encoder']} | {r['pretrain']} | {r['dim']} | " + " | ".join(str(r["test"][c]) for c in COLS) + " |" for r in sel]
            md.append("")
    (out / "report.md").write_text("\n".join(md))
    print("report:", out / "report.md")


if __name__ == "__main__":
    main()
