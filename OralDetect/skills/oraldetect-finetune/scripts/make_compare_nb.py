#!/usr/bin/env python
"""Notebook showing ground truth, before-predictions and after-predictions on the same images.

    python make_compare_nb.py --gt <val.json> --images <data_root> \
        --before <a/preds.bbox.json> --after <b/preds.bbox.json> --out compare.ipynb [--n 8]

Numbers say how much changed; this says what changed. Paths are baked in.
"""
import argparse
import json
import os.path as osp

CELLS = [
    ("md", """# Before / after

Three panels per image: **ground truth** (green), **{lb}** (orange), **{la}** (cyan).
Predictions are drawn above `SCORE`; raise it if the middle and right panels are unreadable.

What to look at:
- boxes that appear on the right and not the middle — what the finetune learned
- boxes that vanish — what it stopped over-predicting
- ground-truth boxes still unmatched in both — what neither model finds
"""),
    ("code", """%matplotlib inline
import json, random, os.path as osp
import matplotlib.pyplot as plt, matplotlib.patches as patches
from PIL import Image
from collections import defaultdict

GT     = {gt!r}
IMAGES = {images!r}
BEFORE = {before!r}
AFTER  = {after!r}
N      = {n}
SEED   = {seed}
SCORE  = {score}
PNG    = {png!r}

gt   = json.load(open(GT))
cats = {{c['id']: c['name'] for c in gt['categories']}}
imgs = {{i['id']: i for i in gt['images']}}
G = defaultdict(list)
for a in gt['annotations']:
    G[a['image_id']].append((a['category_id'], a['bbox']))

def load(p):
    d = defaultdict(list)
    for x in json.load(open(p)):
        if x['score'] >= SCORE:
            d[x['image_id']].append((x['category_id'], x['bbox'], x['score']))
    return d
B, A = load(BEFORE), load(AFTER)
print(f"{{len(gt['images'])}} images, {{len(gt['annotations'])}} GT boxes")
print(f"above score {{SCORE}}:  before {{sum(len(v) for v in B.values()):,}}  "
      f"after {{sum(len(v) for v in A.values()):,}}")"""),
    ("md", "## Samples — ordered by how much the two runs disagree"),
    ("code", """def draw(ax, im, boxes, colour, title, scored):
    p = osp.join(IMAGES, im['file_name'])
    try:
        ax.imshow(Image.open(p).convert('RGB'))
    except Exception as e:
        ax.text(.5, .5, str(e), ha='center'); ax.axis('off'); return
    for b in boxes:
        cid, bb = b[0], b[1]
        x, y, w, h = bb
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, lw=1.5, edgecolor=colour))
        lab = cats.get(cid, '??')
        if scored and len(b) > 2:
            lab += f" {{b[2]:.2f}}"
        if len(boxes) <= 12:
            ax.text(x, max(0, y - 3), lab, fontsize=6.5, color='black',
                    bbox=dict(fc=colour, ec='none', pad=0.5, alpha=.85))
    ax.set_title(f"{{title}} ({{len(boxes)}})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

random.seed(SEED)
ids = [i for i in imgs if G[i]]
# the interesting images are the ones where the two runs differ most
ids.sort(key=lambda i: -abs(len(A[i]) - len(B[i])))
pick = ids[:N] if len(ids) > N else ids

fig, axes = plt.subplots(len(pick), 3, figsize=(15, 4.6 * len(pick)))
axes = axes.reshape(len(pick), 3)
for row, i in enumerate(pick):
    im = imgs[i]
    draw(axes[row][0], im, G[i], 'lime',   'ground truth', False)
    draw(axes[row][1], im, B[i], 'orange', {lb!r},          True)
    draw(axes[row][2], im, A[i], 'cyan',   {la!r},          True)
    axes[row][0].set_ylabel(osp.basename(im['file_name'])[:28], fontsize=7)
plt.tight_layout()
plt.savefig(PNG, dpi=110, bbox_inches='tight')
print('figure saved to', PNG)
plt.show()"""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", default="compare.ipynb")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--label-before", default="before")
    ap.add_argument("--label-after", default="after")
    a = ap.parse_args()
    for p in (a.gt, a.before, a.after):
        if not osp.isfile(p):
            raise SystemExit(f"no such file: {p}")

    fmt = dict(gt=a.gt, images=a.images, before=a.before, after=a.after,
               n=a.n, seed=a.seed, score=a.score, lb=a.label_before,
               la=a.label_after, png=osp.splitext(a.out)[0] + '.png')
    cells = []
    for kind, body in CELLS:
        src = body.format(**fmt)
        cells.append({"cell_type": "markdown" if kind == "md" else "code", "metadata": {},
                      "source": src.splitlines(keepends=True),
                      **({} if kind == "md" else {"execution_count": None, "outputs": []})})

    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        body = "\n".join(l for l in "".join(c["source"]).splitlines()
                         if not l.lstrip().startswith("%"))
        try:
            compile(body, f"<cell {i}>", "exec")
        except SyntaxError as e:
            raise SystemExit(f"BUG in make_compare_nb.py: cell {i} does not compile -- {e}")

    nb = {"cells": cells, "metadata": {"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}  ({a.n} images, score >= {a.score})")


if __name__ == "__main__":
    main()
