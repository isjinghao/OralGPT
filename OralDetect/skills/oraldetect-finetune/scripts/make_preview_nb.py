#!/usr/bin/env python
"""Generate a self-contained notebook that draws random annotated samples from a COCO file.

    python make_preview_nb.py --ann <train.json> --images <data_root> --out preview.ipynb [--n 12]

The point is to see the boxes on the pixels before spending GPU hours. Counts in a json can look
perfect while the boxes sit in the wrong coordinate space, are off by a factor of the image size,
or are xyxy where COCO wants xywh — none of which any structural check can catch, and all of which
train quietly to a bad model.

Paths are baked into the notebook, so it runs anywhere the data is reachable.
"""
import argparse
import json
import os.path as osp


CELLS = [
    ("md", """# Dataset preview

`{ann}`

Look for three things:
1. **Boxes land on the objects.** If they are shifted or scaled, the file is probably xyxy rather
   than COCO's `[x, y, w, h]`, or normalised 0-1 rather than pixels.
2. **Labels match what is in the box.** A systematic off-by-one means the category order and the
   annotations disagree.
3. **Nothing obvious is unlabelled.** A detection ground truth must carry *every* instance on the
   image; missed ones are scored as false positives and actively teach the model to suppress them.
"""),
    ("code", """%matplotlib inline
import json, random, os.path as osp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import Counter, defaultdict

ANN    = {ann!r}
IMAGES = {images!r}
N      = {n}
SEED   = {seed}
PNG    = {png!r}
PNG_CLASSES = PNG.replace('.png', '_classes.png')

d = json.load(open(ANN))
cats = {{c['id']: c['name'] for c in d['categories']}}
by_img = defaultdict(list)
for a in d['annotations']:
    by_img[a['image_id']].append(a)
print(f"{{len(d['images']):,}} images · {{len(d['annotations']):,}} boxes · {{len(cats)}} categories")
print(f"{{sum(1 for i in d['images'] if i['id'] not in by_img):,}} images carry no boxes")"""),
    ("md", "## Class distribution — the long tail decides what is learnable"),
    ("code", """cnt = Counter(cats[a['category_id']] for a in d['annotations'])
order = cnt.most_common()
print(f"{'class':<44}{'boxes':>8}")
for k, v in order:
    print(f"  {k:<42}{v:>8,}")
rare = [k for k, v in order if v < 10]
if rare:
    print(f"\\n!! {len(rare)} classes have <10 boxes: {rare[:8]}")
    print("   They will not train. Consider merging them, or accept they stay near zero AP.")

fig, ax = plt.subplots(figsize=(9, max(3, len(order) * 0.22)))
ax.barh([k for k, _ in order][::-1], [v for _, v in order][::-1])
ax.set_xscale('log'); ax.set_xlabel('boxes (log)'); plt.tight_layout()
plt.savefig(PNG_CLASSES, dpi=110, bbox_inches='tight')
print('figure saved to', PNG_CLASSES)
plt.show()"""),
    ("md", "## Random samples with their boxes"),
    ("code", """random.seed(SEED)
have = [i for i in d['images'] if by_img.get(i['id'])]
pick = random.sample(have, min(N, len(have)))
cols = 3; rows = (len(pick) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows))
for ax, im in zip(axes.ravel(), pick):
    p = osp.join(IMAGES, im['file_name'])
    try:
        img = Image.open(p).convert('RGB')
    except Exception as e:
        ax.text(.5, .5, f"unreadable\\n{im['file_name']}\\n{e}", ha='center', va='center')
        ax.axis('off'); continue
    ax.imshow(img)
    W, H = img.size
    if (W, H) != (im.get('width', W), im.get('height', H)):
        ax.set_title(f"!! json says {im.get('width')}x{im.get('height')}, file is {W}x{H}",
                     color='red', fontsize=8)
    anns = by_img[im['id']]
    # Above ~12 boxes the text overlaps into an unreadable mat and hides the very thing this
    # cell exists to show. Draw every box, but move the names underneath instead of onto it.
    label_on_image = len(anns) <= 12
    for a in anns:
        x, y, w, h = a['bbox']
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, lw=1.4, edgecolor='lime'))
        if label_on_image:
            ax.text(x, max(0, y - 3), cats.get(a['category_id'], '??'), fontsize=7,
                    color='black', bbox=dict(fc='lime', ec='none', pad=0.6, alpha=.85))
    cap = f"{osp.basename(im['file_name'])}  ({len(anns)} boxes)"
    if not label_on_image:
        names = [cats.get(a['category_id'], '??') for a in anns]
        top = Counter(names).most_common(3)
        more = len(set(names)) - len(top)
        line = ", ".join(f"{k[:26]} x{v}" for k, v in top)
        cap += chr(10) + line + (f" (+{more} more classes)" if more > 0 else "")
    ax.set_xlabel(cap, fontsize=6.5)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes.ravel()[len(pick):]:
    ax.axis('off')
plt.tight_layout()
plt.savefig(PNG, dpi=110, bbox_inches='tight')
print('figure saved to', PNG)
plt.show()"""),
    ("md", "## Box geometry — catches the wrong coordinate convention"),
    ("code", """dims = {i['id']: (i.get('width'), i.get('height')) for i in d['images']}
oob = norm = neg = 0
for a in d['annotations']:
    x, y, w, h = a['bbox']
    W, H = dims.get(a['image_id'], (None, None))
    if w <= 0 or h <= 0: neg += 1
    if W and H and (x + w > W * 1.02 or y + h > H * 1.02): oob += 1
    if 0 < w <= 1.0 and 0 < h <= 1.0: norm += 1
n = len(d['annotations'])
def verdict(k, tot, msg):
    r = k / max(tot, 1)
    if r > 0.20:  return f"   <-- {msg}"
    if k:         return f"   ({r*100:.2f}% -- stray annotations, not a format problem)"
    return ""
print(f"degenerate (w or h <= 0)      {neg:>8,}{verdict(neg, n, 'the loader drops all of these')}")
print(f"extending past the image      {oob:>8,}{verdict(oob, n, 'boxes are probably xyxy, not xywh')}")
print(f"w and h both <= 1.0           {norm:>8,}{verdict(norm, n, 'boxes are probably normalised 0-1')}")
print(f"total                         {n:>8,}")
if max(oob, norm) > n * 0.20:
    print("\\n!! STOP and fix the coordinate format before training.")
elif neg or oob or norm:
    print("\\nSmall counts here are normal: a handful of sloppy boxes, not a wrong convention.")""")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="preview.ipynb")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for p in (a.ann,):
        if not osp.isfile(p):
            raise SystemExit(f"no such file: {p}")

    cells = []
    for kind, body in CELLS:
        src = (body.format(ann=a.ann, images=a.images, n=a.n, seed=a.seed,
                          png=osp.splitext(a.out)[0] + ".png")
               if ("{ann" in body or "{images" in body) else body)
        cells.append({
            "cell_type": "markdown" if kind == "md" else "code",
            "metadata": {},
            "source": src.splitlines(keepends=True),
            **({} if kind == "md" else {"execution_count": None, "outputs": []}),
        })
    nb = {"cells": cells, "metadata": {"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}
    # Never emit a notebook that cannot run. The cell bodies live inside triple-quoted strings
    # here, so one mis-escaped backslash silently produces a broken cell that only surfaces when
    # someone executes it.
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        body = "".join(c["source"])
        body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("%"))
        try:
            compile(body, f"<cell {i}>", "exec")
        except SyntaxError as e:
            raise SystemExit(f"BUG in make_preview_nb.py: generated cell {i} does not compile "
                             f"-- {e}. Fix the template, do not ship this notebook.")

    json.dump(nb, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}  ({len(cells)} cells, {a.n} samples, seed {a.seed})")
    print("run it with:  jupyter nbconvert --execute --to notebook --inplace " + a.out)


if __name__ == "__main__":
    main()
