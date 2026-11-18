#!/usr/bin/env python
"""Step 1 -- build every encoder once on CPU so hub weights land in the cache, and check each forward pass.

    python scripts/01_fetch_weights.py [--only vit_b_dinov3 ...]
"""
import argparse
import traceback

import torch

import _bootstrap  # noqa: F401
from encprobe.config import all_encoder_cfgs, load_encoder_cfg
from encprobe.encoders import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    failed = []
    for f in all_encoder_cfgs():
        if args.only and f.stem not in args.only:
            continue
        try:
            cfg = load_encoder_cfg(f)
            m = build(cfg, 1024)
            with torch.no_grad():
                y = m(torch.rand(1, 3, 1024, 1024))
            n = sum(p.numel() for p in m.parameters()) / 1e6
            print(f"OK   {f.stem:24s} params {n:6.1f}M  feature map {tuple(y.shape)}", flush=True)
        except Exception:
            failed.append(f.stem)
            print(f"FAIL {f.stem}\n{traceback.format_exc()}", flush=True)
    raise SystemExit(f"failed: {failed}" if failed else 0)


if __name__ == "__main__":
    main()
