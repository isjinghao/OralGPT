"""Tooth-ID metrics. mAP folds "missed the tooth" and "found it, wrong number" into one figure;
these keep them apart.

    det_recall     found teeth / GT teeth                       (number ignored)
    det_precision  found teeth / predicted teeth                (number ignored)
    id_accuracy    correct number / found teeth
    e2e_recall     found AND correct number / GT teeth          = det_recall * id_accuracy
"""
from __future__ import annotations

from collections import Counter, defaultdict

from ..fdi import FDI_CODES, RELATIONS, relation
from .matching import greedy_match


class IDAccumulator:
    def __init__(self, iou_thr: float = 0.5):
        self.iou_thr = iou_thr
        self.n = defaultdict(Counter)           # group -> {gt, pred, found, correct}
        self.rel = Counter()                    # relation -> count, over found teeth
        self.confusion = Counter()              # (gt code, pred code) -> count
        self.per_code = defaultdict(Counter)    # gt code -> {gt, found, correct}

    def add(self, view: str, gt_codes, gt_boxes, pred_codes, pred_boxes) -> list[tuple[int, int, float]]:
        pairs = greedy_match(gt_boxes, pred_boxes, self.iou_thr) if len(gt_boxes) and len(pred_boxes) else []
        correct = sum(gt_codes[g] == pred_codes[p] for g, p, _ in pairs)
        for grp in ("all", view):
            self.n[grp].update(gt=len(gt_codes), pred=len(pred_codes), found=len(pairs), correct=correct)
        for c in gt_codes:
            self.per_code[c]["gt"] += 1
        for g, p, _ in pairs:
            gc, pc = gt_codes[g], pred_codes[p]
            self.rel[relation(gc, pc)] += 1
            self.confusion[(gc, pc)] += 1
            self.per_code[gc].update(found=1, correct=int(gc == pc))
        return pairs

    @staticmethod
    def _rates(c: Counter) -> dict:
        div = lambda a, b: round(a / b, 4) if b else None
        return dict(n_gt=c["gt"], n_pred=c["pred"], n_found=c["found"], n_correct=c["correct"],
                    det_recall=div(c["found"], c["gt"]), det_precision=div(c["found"], c["pred"]),
                    id_accuracy=div(c["correct"], c["found"]), e2e_recall=div(c["correct"], c["gt"]))

    def summary(self) -> dict:
        found = self.n["all"]["found"]
        wrong = [(k, v) for k, v in self.confusion.most_common() if k[0] != k[1]]
        return dict(
            iou_thr=self.iou_thr,
            overall=self._rates(self.n["all"]),
            per_view={v: self._rates(c) for v, c in sorted(self.n.items()) if v != "all"},
            relations={r: dict(n=self.rel[r], frac=round(self.rel[r] / found, 4) if found else None)
                       for r in RELATIONS},
            per_code={str(c): self._rates(self.per_code[c]) for c in FDI_CODES},
            top_confusions=[dict(gt=g, pred=p, n=n) for (g, p), n in wrong[:20]],
        )


def to_markdown(s: dict, title: str) -> str:
    cols = ("n_gt", "det_recall", "det_precision", "id_accuracy", "e2e_recall")
    row = lambda name, r: f"| {name} | " + " | ".join(str(r[c]) for c in cols) + " |"
    out = [f"# {title}", "", f"Matching: class-agnostic, IoU >= {s['iou_thr']}.", "",
           "| group | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols), row("all", s["overall"])]
    out += [row(v, r) for v, r in s["per_view"].items()]
    out += ["", "## Where the number goes wrong (share of found teeth)", "", "| relation | n | frac |", "|---|---|---|"]
    out += [f"| {r} | {d['n']} | {d['frac']} |" for r, d in s["relations"].items()]
    out += ["", "## Per FDI code", "", "| code | n_gt | det_recall | id_accuracy | e2e_recall |", "|---|---|---|---|---|"]
    out += [f"| {c} | {r['n_gt']} | {r['det_recall']} | {r['id_accuracy']} | {r['e2e_recall']} |"
            for c, r in s["per_code"].items()]
    out += ["", "## Top confusions", "", "| gt | pred | n |", "|---|---|---|"]
    out += [f"| {d['gt']} | {d['pred']} | {d['n']} |" for d in s["top_confusions"]]
    return "\n".join(out) + "\n"
