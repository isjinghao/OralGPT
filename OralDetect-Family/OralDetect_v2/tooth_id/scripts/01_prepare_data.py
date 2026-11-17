#!/usr/bin/env python
"""Step 1 -- raw IOP-Compass -> YOLO-seg dataset (resized JPEGs, labels, patient-level split).

    python scripts/01_prepare_data.py --data configs/data/iop_compass.yaml [--workers 16]
"""
import argparse
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import _bootstrap  # noqa: F401
from toothid.config import load_data_cfg
from toothid.data.iop_compass import list_records
from toothid.data.split import split_patients
from toothid.data.yolo_format import write_dataset_yaml, write_sample


def _job(a):
    return write_sample(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_data_cfg(args.data)
    root = Path(cfg["yolo_root"])

    recs = list_records(cfg["raw_root"])
    split = split_patients([r.patient for r in recs], cfg["split"]["ratios"], cfg["split"]["seed"])
    names = list(cfg["split"]["ratios"])
    for s in names:
        (root / "images" / s).mkdir(parents=True, exist_ok=True)
        (root / "labels" / s).mkdir(parents=True, exist_ok=True)

    jobs = [(r, split[r.patient], root, cfg["image"]["max_side"], cfg["image"]["jpeg_quality"]) for r in recs]
    with Pool(args.workers) as pool:
        manifest = pool.map(_job, jobs, chunksize=8)

    (root / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    (root / "splits.json").write_text(json.dumps(
        {s: sorted((p for p, v in split.items() if v == s), key=int) for s in names}, indent=1))
    yaml_path = write_dataset_yaml(root, names)

    for s in names:
        ms = [m for m in manifest if m["split"] == s]
        print(f"{s:5s} patients={len({m['patient'] for m in ms}):4d} images={len(ms):5d} "
              f"teeth={sum(m['n_teeth'] for m in ms):6d} views={dict(Counter(m['view'] for m in ms))}")
    print("dataset yaml:", yaml_path)


if __name__ == "__main__":
    main()
