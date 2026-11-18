#!/usr/bin/env python
"""Step 6 -- put the overlays of one folder on a single contact sheet.

    python scripts/06_montage.py --dir <folder of overlays> --out <sheet.jpg> [--glob 'IO150K__*.jpg']
        [--order index.tsv] [--title ...] [--cols 4] [--strip IO150K__]

--order: a TSV whose first column lists stems in the wanted order (the index.tsv of 04_visualize --worst);
without it files are sorted by name.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from toothid.viz.montage import montage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--order")
    ap.add_argument("--title")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--strip", default="", help="text removed from the captions")
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob(args.glob))
    if args.order:
        rank = {l.split("\t")[0]: i for i, l in enumerate(Path(args.order).read_text().splitlines()[1:])}
        files.sort(key=lambda p: rank.get(p.stem.replace(args.strip, ""), len(rank)))
    print("wrote", montage(files, args.out, args.cols, title=args.title, strip=args.strip), "from", len(files), "images")


if __name__ == "__main__":
    main()
