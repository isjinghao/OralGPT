#!/usr/bin/env python
"""Step 2 -- train.

    python scripts/02_train.py --data configs/data/iop_compass.yaml --model configs/model/yolo11m_seg.yaml
    ... --set epochs=1 fraction=0.05 --name smoke      # smoke test; --set overrides `train:` keys
"""
import argparse
from pathlib import Path

import yaml

import _bootstrap  # noqa: F401
from toothid.config import load_data_cfg, load_model_cfg, run_dir
from toothid.engine.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", help="override the run name from the model yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    data, model = load_data_cfg(args.data), load_model_cfg(args.model)
    if args.name:
        model["name"] = args.name
    overrides = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    train(model, Path(data["yolo_root"]) / "dataset.yaml", overrides)

    rd = run_dir(model)                      # keep the exact recipe next to the weights
    model["train"].update(overrides)
    (rd / "resolved_config.yaml").write_text(yaml.safe_dump(dict(data=data, model=model), sort_keys=False))
    print("run dir:", rd)


if __name__ == "__main__":
    main()
