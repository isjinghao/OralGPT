#!/usr/bin/env python
"""Step 4 -- data-scaling test: fine-tune one encoder end to end at every training-set fraction.

    python scripts/04_finetune.py --encoder configs/encoders/vit_b_dinov3.yaml \
        --probe configs/probe/tooth_fdi_finetune.yaml [--fractions 0.01 1.0]

Writes <out_root>/<probe>/runs/<encoder>_f<fraction>.json (skipped if present).
"""
import argparse
import json
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from encprobe.config import load_encoder_cfg, load_probe_cfg
from encprobe.encoders import build
from encprobe.finetune import finetune


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--fractions", nargs="*", type=float)
    args = ap.parse_args()
    enc, cfg = load_encoder_cfg(args.encoder), load_probe_cfg(args.probe)
    runs = Path(cfg["out_root"]) / cfg["name"] / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for frac in args.fractions or cfg["fractions"]:
        out = runs / f"{enc['name']}_f{frac}.json"
        if out.exists():
            print("exists, skipping:", out)
            continue
        print(f"== {enc['name']} fraction {frac}", flush=True)
        encoder = build(enc, cfg["image_size"])           # fresh pretrained weights for every run
        with torch.no_grad():
            dim = encoder(torch.zeros(1, 3, 64, 64) if enc["backbone"].startswith("convnext") else
                          torch.zeros(1, 3, cfg["image_size"], cfg["image_size"])).shape[1]
        r = finetune(encoder, dim, cfg, frac)
        out.write_text(json.dumps(dict(encoder=enc["name"], backbone=enc["backbone"], pretrain=enc["pretrain"], **r), indent=1))
        print(f"   best val {r['val_fdi']} at step {r['step']}: test fdi {r['test']['fdi']} ({r['minutes']} min)", flush=True)
        del encoder
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
