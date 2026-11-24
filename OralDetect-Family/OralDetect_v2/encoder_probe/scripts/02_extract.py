#!/usr/bin/env python
"""Step 2 -- cache one feature vector per ground-truth box, for one encoder at every probe resolution.

    python scripts/02_extract.py --encoder configs/encoders/vit_b_dinov3.yaml --probe configs/probe/tooth_fdi.yaml

Writes <out_root>/<probe>/features/<encoder>_<size>.pt = {train, val, test: {feat, geom, cls, view | mod}}
(+ `classes` for a condition probe).
"""
import argparse
import time

import torch

import _bootstrap  # noqa: F401
from encprobe.config import feature_file, load_encoder_cfg, load_probe_cfg
from encprobe.encoders import build
from encprobe.extract import extract_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    enc, probe = load_encoder_cfg(args.encoder), load_probe_cfg(args.probe)
    for size in probe["image_sizes"]:
        out = feature_file(probe, enc["name"], size)
        if out.exists() and not args.force:
            print("exists, skipping:", out)
            continue
        try:
            model = build(enc, size).cuda()
        except ValueError as e:                      # e.g. SAM runs at its native size only
            print(f"skip {enc['name']} @ {size}: {e}")
            continue
        t0 = time.time()
        data = extract_all(model, probe, size)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(encoder=enc, size=size, params_m=round(sum(p.numel() for p in model.parameters()) / 1e6, 1), **data), out)
        print(f"{enc['name']} @ {size}: {tuple(data['train']['feat'].shape)} train boxes, {time.time() - t0:.0f}s -> {out}", flush=True)
        del model


if __name__ == "__main__":
    main()
