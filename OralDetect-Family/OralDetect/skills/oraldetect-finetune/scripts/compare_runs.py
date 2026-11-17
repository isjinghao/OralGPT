#!/usr/bin/env python
"""Score two prediction files against the same ground truth: precision / recall / F1, plus mAP.

    python compare_runs.py --gt <val.json> --before <a/preds.bbox.json> --after <b/preds.bbox.json>

Detection has no true negatives, so there is no meaningful accuracy. What is reported instead:

  TP   a prediction matched an unmatched GT box of the same class at IoU >= --iou
  FP   a prediction that matched nothing
  FN   a GT box that no prediction matched
  P = TP/(TP+FP)   R = TP/(TP+FN)   F1 = 2PR/(P+R)

P/R/F1 depend on the score threshold, so both are reported: the value at --score, and the best F1
over a sweep of thresholds with the threshold that produced it. Comparing at a single arbitrary
threshold can invert the ranking of two models; the sweep says whether the difference is real.
"""
import argparse
import json
from collections import defaultdict


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def match(gt_by, preds, thr_iou, thr_score):
    """greedy highest-score-first matching, per (image, category)"""
    per = defaultdict(lambda: [0, 0, 0])          # cat -> [TP, FP, FN]
    used = defaultdict(set)
    kept = [p for p in preds if p["score"] >= thr_score]
    for p in sorted(kept, key=lambda p: -p["score"]):
        key = (p["image_id"], p["category_id"])
        best, bj = thr_iou, None
        for j, g in enumerate(gt_by.get(key, [])):
            if j in used[key]:
                continue
            v = iou(p["bbox"], g)
            if v >= best:
                best, bj = v, j
        if bj is None:
            per[p["category_id"]][1] += 1          # FP
        else:
            used[key].add(bj)
            per[p["category_id"]][0] += 1          # TP
    for (img, cat), boxes in gt_by.items():
        per[cat][2] += len(boxes) - len(used[(img, cat)])   # FN
    return per


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def sweep(gt_by, preds, thr_iou, grid):
    best = (0.0, 0.0, 0.0, 0.0)                    # f1, p, r, thr
    for t in grid:
        per = match(gt_by, preds, thr_iou, t)
        tp = sum(v[0] for v in per.values())
        fp = sum(v[1] for v in per.values())
        fn = sum(v[2] for v in per.values())
        p, r, f = prf(tp, fp, fn)
        if f > best[0]:
            best = (f, p, r, t)
    return best


def mean_ap(gt_path, pred_path):
    import contextlib, io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO(gt_path)
        pr = json.load(open(pred_path))
        if not pr:
            return 0.0
        e = COCOeval(g, g.loadRes(pr), "bbox")
        e.evaluate(); e.accumulate(); e.summarize()
    return float(e.stats[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--score", type=float, default=0.3, help="threshold for the fixed-point table")
    ap.add_argument("--label-before", default="before")
    ap.add_argument("--label-after", default="after")
    ap.add_argument("--save", help="also write the report to this file")
    a = ap.parse_args()

    import builtins
    _sink = []
    _real = builtins.print
    def print(*args, **kw):          # noqa: A001 -- tee everything to the report file
        _sink.append(" ".join(str(x) for x in args))
        _real(*args, **kw)

    gt = json.load(open(a.gt))
    name = {c["id"]: c["name"] for c in gt["categories"]}
    gt_by = defaultdict(list)
    for x in gt["annotations"]:
        gt_by[(x["image_id"], x["category_id"])].append(x["bbox"])
    ngt = {c: 0 for c in name}
    for (_, c), b in gt_by.items():
        ngt[c] += len(b)

    runs = {}
    for tag, path in ((a.label_before, a.before), (a.label_after, a.after)):
        preds = json.load(open(path))
        runs[tag] = {"preds": preds,
                     "fixed": match(gt_by, preds, a.iou, a.score),
                     "best": sweep(gt_by, preds, a.iou,
                                   [i / 100 for i in range(5, 96, 5)]),
                     "map": mean_ap(a.gt, path)}
        print(f"  {tag:<12} {len(preds):>7,} raw predictions")

    L, R = a.label_before, a.label_after
    cats = [c for c in name if ngt[c] > 0]

    print()
    print("=" * 105)
    print(f"per class @ IoU {a.iou}, score {a.score}")
    print("=" * 105)
    print(f"{'':<35}{L[:14]:^33} | {R[:14]:^33}")
    print(f"{'class':<30}{'GT':>5}"
          + "".join(f"{h:>11}" for h in ("P", "R", "F1"))
          + " | " + "".join(f"{h:>11}" for h in ("P", "R", "F1")))
    print("-" * 105)
    for c in sorted(cats, key=lambda c: -ngt[c]):
        row = f"{name[c][:28]:<30}{ngt[c]:>5}"
        for j, tag in enumerate((L, R)):
            tp, fp, fn = runs[tag]["fixed"][c]
            p, r, f = prf(tp, fp, fn)
            row += ("" if j == 0 else " | ") + f"{p:>11.3f}{r:>11.3f}{f:>11.3f}"
        print(row)
    print("-" * 105)

    print()
    print("=" * 78)
    print(f"{'metric':<30}{L:>14}{R:>14}{'delta':>14}")
    print("-" * 78)
    tot = {}
    for tag in (L, R):
        per = runs[tag]["fixed"]
        tp = sum(v[0] for v in per.values()); fp = sum(v[1] for v in per.values())
        fn = sum(v[2] for v in per.values())
        tot[tag] = (tp, fp, fn) + prf(tp, fp, fn)
    for i, lab in enumerate(("TP", "FP", "FN")):
        print(f"{lab + f'  @score {a.score}':<30}{tot[L][i]:>14,}{tot[R][i]:>14,}"
              f"{tot[R][i] - tot[L][i]:>+14,}")
    for i, lab in zip((3, 4, 5), ("precision", "recall", "F1")):
        d = tot[R][i] - tot[L][i]
        print(f"{lab + f'  @score {a.score}':<30}{tot[L][i]:>14.4f}{tot[R][i]:>14.4f}{d:>+14.4f}"
              f"{'' if abs(d) < 0.005 else ('  ↑' if d > 0 else '  ↓')}")
    print("-" * 78)
    for i, lab in zip((1, 2, 0), ("precision @bestF1", "recall @bestF1", "best F1")):
        vb, va = runs[L]["best"][i], runs[R]["best"][i]
        d = va - vb
        print(f"{lab:<30}{vb:>14.4f}{va:>14.4f}{d:>+14.4f}"
              f"{'' if abs(d) < 0.005 else ('  ↑' if d > 0 else '  ↓')}")
    print(f"{'  (at score threshold)':<30}{runs[L]['best'][3]:>14.2f}{runs[R]['best'][3]:>14.2f}")
    print("-" * 78)
    d = runs[R]["map"] - runs[L]["map"]
    print(f"{'mAP @[.50:.95]':<30}{runs[L]['map']:>14.4f}{runs[R]['map']:>14.4f}{d:>+14.4f}"
          f"{'' if abs(d) < 0.005 else ('  ↑' if d > 0 else '  ↓')}")
    print("=" * 78)

    # honest reading -- a positive delta on a tiny, near-zero baseline is not a success
    fb, fa = runs[L]["best"][0], runs[R]["best"][0]
    tot_gt = sum(ngt.values())
    if fa <= fb:
        print("\nF1 did not improve. Check: boxes per class, lr matched to the starting checkpoint,")
        print("epochs, and whether the val split is large enough for the difference to mean anything.")
    elif fa < 0.2 or tot_gt < 100:
        print(f"\nF1 improved, but it is {fa:.3f} on {tot_gt} ground-truth boxes. Report the absolute")
        print("value and the data size alongside the delta -- a rise from near-zero to near-zero is")
        print("not evidence the model is usable on this data.")


    if a.save:
        import re as _re
        _ANSI = _re.compile(r"\x1b\[[0-9;]*m")
        with open(a.save, "w") as f:
            f.write(_ANSI.sub("", "\n".join(_sink)) + "\n")
        _real(f"\nreport written to {a.save}")


if __name__ == "__main__":
    raise SystemExit(main())
