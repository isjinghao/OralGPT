"""Class-agnostic one-to-one matching between ground-truth and predicted boxes."""
from __future__ import annotations

import numpy as np


def box_iou(a, b) -> np.ndarray:
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.clip(area(a)[:, None] + area(b)[None] - inter, 1e-9, None)


def greedy_match(gt_boxes, pred_boxes, iou_thr: float) -> list[tuple[int, int, float]]:
    """[(gt_idx, pred_idx, iou)], highest IoU first, each box used at most once. The tooth number
    plays no part: this answers "was the tooth found", so that ID errors can be counted apart."""
    iou = box_iou(gt_boxes, pred_boxes)
    pairs, used_g, used_p = [], set(), set()
    for flat in np.argsort(-iou, axis=None):
        g, p = divmod(int(flat), iou.shape[1])
        if iou[g, p] < iou_thr:
            break
        if g not in used_g and p not in used_p:
            pairs.append((g, p, float(iou[g, p])))
            used_g.add(g), used_p.add(p)
    return pairs
