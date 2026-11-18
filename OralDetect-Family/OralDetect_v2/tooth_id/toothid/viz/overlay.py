"""Prediction overlay: green = right number, red = wrong number (pred/gt), orange = spurious box,
dashed white = missed tooth. Each orange box also names the ground-truth box it overlaps most
(dashed cyan when that box is already matched to another prediction), or says "no gt"."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from ..eval.matching import box_iou, greedy_match

GREEN, RED, ORANGE, WHITE, CYAN = "#2ecc71", "#ff3b30", "#ff9f0a", "#ffffff", "#32d7ff"


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _dashed_rect(dr, box, color, width, dash=10):
    x1, y1, x2, y2 = box
    for a, b, horiz in ((x1, x2, True), (y1, y2, False)):
        for s in range(int(a), int(b), dash * 2):
            e = min(s + dash, b)
            for fixed in ((y1, y2) if horiz else (x1, x2)):
                dr.line([(s, fixed), (e, fixed)] if horiz else [(fixed, s), (fixed, e)], fill=color, width=width)


def draw(image_path, gt_codes, gt_boxes, pred_codes, pred_boxes, iou_thr: float = 0.5, pred_scores=None) -> Image.Image:
    im = Image.open(image_path).convert("RGB")
    dr, width = ImageDraw.Draw(im), max(2, im.width // 500)
    font = _font(max(14, im.width // 60))
    pairs = greedy_match(gt_boxes, pred_boxes, iou_thr) if len(gt_boxes) and len(pred_boxes) else []
    hit_g, hit_p = {g for g, _, _ in pairs}, {p: g for g, p, _ in pairs}
    iou = box_iou(gt_boxes, pred_boxes) if len(gt_boxes) and len(pred_boxes) else None
    for g, box in enumerate(gt_boxes):
        if g not in hit_g:
            _dashed_rect(dr, box, WHITE, width)
            dr.text((box[0] + 3, box[1] + 3), f"miss {gt_codes[g]}", fill=WHITE, font=font, stroke_width=2, stroke_fill="black")
    for p, box in enumerate(pred_boxes):
        if p not in hit_p:
            color, text, near = ORANGE, f"{pred_codes[p]}?", "no gt"
            g = int(iou[:, p].argmax()) if iou is not None else -1
            if g >= 0 and iou[g, p] > 0:     # the ground truth this spurious box sits on
                near = f"gt{gt_codes[g]} iou" + f"{iou[g, p]:.2f}".replace("0.", ".", 1)
                if g in hit_g:               # unmatched ones are already drawn as a miss
                    _dashed_rect(dr, gt_boxes[g], CYAN, width)
        elif gt_codes[hit_p[p]] == pred_codes[p]:
            color, text = GREEN, str(pred_codes[p])
        else:
            color, text = RED, f"{pred_codes[p]}/{gt_codes[hit_p[p]]}"
        if pred_scores is not None:      # confidence on a second line, e.g. ".87"
            text += f"\n{pred_scores[p]:.2f}".replace("0.", ".", 1)
        if p not in hit_p:
            text += f"\n{near}"
        dr.rectangle(box, outline=color, width=width)
        dr.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), text, fill=color, font=font, anchor="mm", align="center",
                stroke_width=2, stroke_fill="black")
    return im
