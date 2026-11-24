"""Full FDI numbers for sets with human tooth boxes but incomplete numbers.

tooth_type   (IO150K)  the type is labelled, the quadrant is not: borrow the quadrant from a model.
anterior12   (FDTooth) the 12 anterior teeth are boxed, unnumbered: number them from geometry, and
             take the other teeth of the photo from a model so they are not trained as background.
Images where this cannot be done cleanly are dropped, not guessed.
"""
from __future__ import annotations

from ..eval.matching import greedy_match
from ..fdi import quadrant


def complete_codes(positions, gt_boxes, pred_codes, pred_boxes, iou_thr: float) -> list[int] | None:
    """-> one FDI code per ground-truth box, or None when the image should be dropped.

    A ground-truth tooth matched to a predicted box takes that prediction's quadrant. An unmatched
    one takes the quadrant of the nearest numbered tooth whose position is one off (its neighbour
    in the same quadrant). None if a tooth has no such neighbour or a code ends up used twice.
    """
    quad = [None] * len(positions)
    for g, p, _ in greedy_match(gt_boxes, pred_boxes, iou_thr) if len(pred_boxes) else []:
        quad[g] = quadrant(pred_codes[p])
    centre = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in gt_boxes]
    known = [i for i, q in enumerate(quad) if q is not None]
    for i, q in enumerate(quad):
        if q is not None:
            continue
        near = [j for j in known if abs(positions[j] - positions[i]) == 1]
        if not near:
            return None
        j = min(near, key=lambda j: (centre[j][0] - centre[i][0]) ** 2 + (centre[j][1] - centre[i][1]) ** 2)
        quad[i] = quad[j]
    codes = [q * 10 + p for q, p in zip(quad, positions)]
    return codes if len(set(codes)) == len(codes) else None


def box_label_line(class_idx: int, box, w: int, h: int) -> str:
    """A box as the 4-corner polygon line of the YOLO-seg label format used by this module."""
    x1, y1, x2, y2 = (min(max(v / s, 0), 1) for v, s in zip(box, (w, h, w, h)))
    return f"{class_idx} " + " ".join(f"{v:.6f}" for v in (x1, y1, x2, y1, x2, y2, x1, y2))


ANTERIOR_UPPER = (13, 12, 11, 21, 22, 23)      # left to right in a frontal photo (image left = patient's right)
ANTERIOR_LOWER = (43, 42, 41, 31, 32, 33)


def anterior12_codes(boxes) -> list[int] | None:
    """FDI codes for a frontal photo whose boxes are exactly the 12 anterior teeth (canine to canine,
    both arches), from geometry alone: the 6 higher boxes are the upper arch, each arch is read left
    to right. None when there are not 12 boxes."""
    if len(boxes) != 12:
        return None
    by_y = sorted(range(12), key=lambda i: boxes[i][1] + boxes[i][3])
    codes = [0] * 12
    for idx, names in ((by_y[:6], ANTERIOR_UPPER), (by_y[6:], ANTERIOR_LOWER)):
        for i, c in zip(sorted(idx, key=lambda i: boxes[i][0] + boxes[i][2]), names):
            codes[i] = c
    return codes


def other_teeth(gt_boxes, gt_codes, pred_codes, pred_boxes, pred_scores, iou_thr: float, min_score: float):
    """Teeth the ground truth leaves unlabelled, taken from a model's predictions so that they are
    not trained as background. -> (codes, boxes), or None when the model contradicts the ground truth.

    A prediction overlapping a ground-truth box (IoU >= iou_thr) must carry that box's code, else the
    image is rejected. Predictions clear of every ground-truth box are kept when confident, numbered
    with a code the image does not have yet.
    """
    from ..eval.matching import box_iou

    if not len(pred_boxes):
        return [], []
    iou = box_iou(gt_boxes, pred_boxes)
    for g, p, _ in greedy_match(gt_boxes, pred_boxes, iou_thr):
        if pred_codes[p] != gt_codes[g]:
            return None
    codes, boxes, seen = [], [], set(gt_codes)
    for p in sorted(range(len(pred_boxes)), key=lambda p: -pred_scores[p]):
        if iou[:, p].max() < iou_thr and pred_scores[p] >= min_score and pred_codes[p] not in seen:
            seen.add(pred_codes[p])
            codes.append(pred_codes[p]), boxes.append(pred_boxes[p])
    return codes, boxes
