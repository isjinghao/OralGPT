"""Metrics when the ground truth gives the tooth type but not the quadrant: only the position digit
of the predicted FDI code can be scored.

    det_recall / det_precision   as in id_metrics (number ignored)
    type_accuracy                right position digit / found teeth
    e2e_recall                   found AND right position / GT teeth
"""
from __future__ import annotations

from collections import Counter, defaultdict

from ..fdi import POSITION_NAMES, position
from .matching import greedy_match


class TypeAccumulator:
    def __init__(self, iou_thr: float = 0.5):
        self.iou_thr = iou_thr
        self.n = defaultdict(Counter)           # group -> {gt, pred, found, correct}
        self.per_pos = defaultdict(Counter)     # gt position -> {gt, found, correct}
        self.confusion = Counter()              # (gt position, pred position) -> count
        self.offset = Counter()                 # pred position - gt position, over found teeth

    def add(self, group: str, gt_pos, gt_boxes, pred_codes, pred_boxes) -> list[tuple[int, int, float]]:
        pairs = greedy_match(gt_boxes, pred_boxes, self.iou_thr) if len(gt_boxes) and len(pred_boxes) else []
        correct = sum(gt_pos[g] == position(pred_codes[p]) for g, p, _ in pairs)
        for grp in ("all", group):
            self.n[grp].update(gt=len(gt_pos), pred=len(pred_codes), found=len(pairs), correct=correct)
        for q in gt_pos:
            self.per_pos[q]["gt"] += 1
        for g, p, _ in pairs:
            gq, pq = gt_pos[g], position(pred_codes[p])
            self.confusion[(gq, pq)] += 1
            self.offset[pq - gq] += 1
            self.per_pos[gq].update(found=1, correct=int(gq == pq))
        return pairs

    @staticmethod
    def _rates(c: Counter) -> dict:
        div = lambda a, b: round(a / b, 4) if b else None
        return dict(n_gt=c["gt"], n_pred=c["pred"], n_found=c["found"], n_correct=c["correct"],
                    det_recall=div(c["found"], c["gt"]), det_precision=div(c["found"], c["pred"]),
                    type_accuracy=div(c["correct"], c["found"]), e2e_recall=div(c["correct"], c["gt"]))

    def summary(self) -> dict:
        found = self.n["all"]["found"]
        wrong = [(k, v) for k, v in self.confusion.most_common() if k[0] != k[1]]
        return dict(
            iou_thr=self.iou_thr,
            overall=self._rates(self.n["all"]),
            per_group={g: self._rates(c) for g, c in sorted(self.n.items()) if g != "all"},
            per_type={POSITION_NAMES[q]: self._rates(self.per_pos[q]) for q in sorted(self.per_pos)},
            offset={str(k): dict(n=v, frac=round(v / found, 4)) for k, v in sorted(self.offset.items())},
            top_confusions=[dict(gt=POSITION_NAMES[g], pred=POSITION_NAMES[p], n=n) for (g, p), n in wrong[:15]],
        )


def to_markdown(s: dict, title: str) -> str:
    cols = ("n_gt", "det_recall", "det_precision", "type_accuracy", "e2e_recall")
    row = lambda name, r: f"| {name} | " + " | ".join(str(r[c]) for c in cols) + " |"
    head = ["| group | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    out = [f"# {title}", "", f"Matching: class-agnostic, IoU >= {s['iou_thr']}. Ground truth has the tooth type only, "
           "so a prediction is right when its position digit is right; the quadrant is not scored.", ""]
    out += head + [row("all", s["overall"])] + [row(g, r) for g, r in s["per_group"].items()]
    out += ["", "## Per tooth type", ""] + head + [row(t, r) for t, r in s["per_type"].items()]
    out += ["", "## Predicted position minus true position (share of found teeth)", "", "| offset | n | frac |", "|---|---|---|"]
    out += [f"| {k} | {d['n']} | {d['frac']} |" for k, d in s["offset"].items()]
    out += ["", "## Top confusions", "", "| gt | pred | n |", "|---|---|---|"]
    out += [f"| {d['gt']} | {d['pred']} | {d['n']} |" for d in s["top_confusions"]]
    return "\n".join(out) + "\n"
