#!/usr/bin/env python
"""Detect what annotation format a folder holds, and convert it to COCO detection json.

    python to_coco.py --root <folder>                      # detect only, report what it is
    python to_coco.py --root <folder> --convert --out <dir> [--val-frac 0.1] [--seed 0]

Handles COCO (pass-through), YOLO (.txt), Pascal VOC (.xml), LabelMe (per-image .json).
Anything else is reported, not guessed at.
"""
import argparse
import json
import os
import os.path as osp
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def walk(root, exts=None, cap=200000):
    out = []
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            if exts is None or f.lower().endswith(exts):
                out.append(osp.join(dp, f))
                if len(out) >= cap:
                    return out
    return out


def imsize(p):
    from PIL import Image
    with Image.open(p) as im:
        return im.size


# ------------------------------------------------------------------ detection
def detect(root):
    imgs = walk(root, IMG_EXT)
    jsons = walk(root, (".json",))
    txts = [p for p in walk(root, (".txt",))
            if osp.basename(p).lower() not in ("classes.txt", "obj.names", "readme.txt")]
    xmls = walk(root, (".xml",))

    for p in jsons:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if isinstance(d, dict) and {"images", "annotations", "categories"} <= set(d):
            return "coco", {"files": [p], "images": imgs}
    labelme = [p for p in jsons
               if _try(p, lambda d: isinstance(d, dict) and "shapes" in d and "imagePath" in d)]
    if len(labelme) >= max(1, len(imgs) * 0.3):
        return "labelme", {"files": labelme, "images": imgs}
    voc = [p for p in xmls if _try_xml(p)]
    if len(voc) >= max(1, len(imgs) * 0.3):
        return "voc", {"files": voc, "images": imgs}
    if txts and len(txts) >= max(1, len(imgs) * 0.3):
        return "yolo", {"files": txts, "images": imgs}
    return "unknown", {"files": [], "images": imgs,
                       "counts": {"json": len(jsons), "txt": len(txts), "xml": len(xmls)}}


def _try(p, fn):
    try:
        return fn(json.load(open(p)))
    except Exception:
        return False


def _try_xml(p):
    try:
        r = ET.parse(p).getroot()
        return r.tag == "annotation" and r.find("object") is not None
    except Exception:
        return False


# ------------------------------------------------------------------ converters
def _stem_map(images):
    m = {}
    for p in images:
        m.setdefault(osp.splitext(osp.basename(p))[0], p)
    return m


def from_yolo(root, files, images):
    names = None
    for cand in ("classes.txt", "obj.names"):
        hits = [p for p in walk(root, (cand[-4:],)) if osp.basename(p).lower() == cand]
        if hits:
            names = [l.strip() for l in open(hits[0]) if l.strip()]
            break
    if names is None:
        for y in walk(root, (".yaml", ".yml")):
            t = open(y, errors="ignore").read()
            m = re.search(r"names\s*:\s*\[(.*?)\]", t, re.S)
            if m:
                names = [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
                break
            m = re.search(r"names\s*:\s*\n((?:\s*-\s*.+\n?)+)", t)
            if m:
                names = [l.split("-", 1)[1].strip().strip("'\"") for l in m.group(1).splitlines()
                         if l.strip()]
                break
    if names is None:
        raise SystemExit("YOLO layout but no class-name file. Expected classes.txt, obj.names, or "
                         "a data.yaml with `names:`. Ask the user which id means what.")

    smap = _stem_map(images)
    recs = []
    for t in files:
        stem = osp.splitext(osp.basename(t))[0]
        img = smap.get(stem)
        if not img:
            continue
        W, H = imsize(img)
        boxes = []
        for line in open(t):
            f = line.split()
            if len(f) < 5:
                continue
            c, cx, cy, w, h = int(float(f[0])), *map(float, f[1:5])
            # YOLO is normalised centre-form; COCO wants absolute top-left + size
            boxes.append((c, (cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H))
        recs.append((img, W, H, boxes))
    return names, recs


def from_voc(root, files, images):
    smap = _stem_map(images)
    names, recs = [], []
    for x in files:
        r = ET.parse(x).getroot()
        stem = osp.splitext(osp.basename(x))[0]
        img = smap.get(stem)
        if not img:
            fn = r.findtext("filename")
            img = smap.get(osp.splitext(fn)[0]) if fn else None
        if not img:
            continue
        sz = r.find("size")
        W = int(sz.findtext("width")) if sz is not None else None
        H = int(sz.findtext("height")) if sz is not None else None
        if not W or not H:
            W, H = imsize(img)
        boxes = []
        for o in r.findall("object"):
            nm = o.findtext("name")
            if nm not in names:
                names.append(nm)
            b = o.find("bndbox")
            x1, y1 = float(b.findtext("xmin")), float(b.findtext("ymin"))
            x2, y2 = float(b.findtext("xmax")), float(b.findtext("ymax"))
            boxes.append((names.index(nm), x1, y1, x2 - x1, y2 - y1))
        recs.append((img, W, H, boxes))
    return names, recs


def from_labelme(root, files, images):
    smap = _stem_map(images)
    names, recs = [], []
    for j in files:
        d = json.load(open(j))
        stem = osp.splitext(osp.basename(d.get("imagePath", j)))[0]
        img = smap.get(stem) or smap.get(osp.splitext(osp.basename(j))[0])
        if not img:
            continue
        W, H = d.get("imageWidth"), d.get("imageHeight")
        if not W or not H:
            W, H = imsize(img)
        boxes = []
        for s in d.get("shapes", []):
            pts = s.get("points") or []
            if len(pts) < 2:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]   # polygons -> their bounding box
            nm = s.get("label")
            if nm not in names:
                names.append(nm)
            boxes.append((names.index(nm), min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))
        recs.append((img, W, H, boxes))
    return names, recs


def write_coco(recs, names, root, out, val_frac, seed):
    random.seed(seed)
    recs = sorted(recs)
    idx = list(range(len(recs)))
    random.shuffle(idx)
    nval = int(len(idx) * val_frac)
    split = {"val": set(idx[:nval]), "train": set(idx[nval:])} if nval else {"train": set(idx)}

    os.makedirs(out, exist_ok=True)
    written = {}
    for name, keep in split.items():
        imgs, anns, aid = [], [], 1
        for i, (p, W, H, boxes) in enumerate(recs):
            if i not in keep:
                continue
            rel = osp.relpath(p, root)
            imgs.append({"id": i, "file_name": rel, "width": W, "height": H})
            for c, x, y, w, h in boxes:
                if w <= 0 or h <= 0:
                    continue
                anns.append({"id": aid, "image_id": i, "category_id": c,
                             "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                             "area": round(w * h, 2), "iscrowd": 0})
                aid += 1
        d = {"images": imgs, "annotations": anns,
             "categories": [{"id": i, "name": n} for i, n in enumerate(names)]}
        p = osp.join(out, f"instances_{name}.json")
        json.dump(d, open(p, "w"))
        written[name] = (p, len(imgs), len(anns))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--split", metavar="COCO_JSON",
                    help="split an existing COCO file into train/val by IMAGE and exit")
    ap.add_argument("--out")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", help="also write this report to a file")
    a = ap.parse_args()

    import builtins as _b
    _sink, _real = [], _b.print

    def print(*args, **kw):          # noqa: A001 -- tee the report to --save
        _sink.append(" ".join(str(x) for x in args))
        _real(*args, **kw)

    if a.save:
        import atexit, os as _os
        import re as _re
        _ANSI = _re.compile(r"\x1b\[[0-9;]*m")

        def _flush():                # runs on every exit path, including SystemExit
            _os.makedirs(_os.path.dirname(_os.path.abspath(a.save)) or ".", exist_ok=True)
            with open(a.save, "w") as _fh:
                _fh.write(_ANSI.sub("", "\n".join(_sink)) + "\n")
        atexit.register(_flush)


    if a.split:
        if not a.out:
            raise SystemExit("--split needs --out")
        d = json.load(open(a.split))
        by_img = {}
        for x in d["annotations"]:
            by_img.setdefault(x["image_id"], []).append(x)
        ids = sorted(i["id"] for i in d["images"])
        random.seed(a.seed)
        random.shuffle(ids)
        nval = max(1, int(len(ids) * a.val_frac))
        parts = {"val": set(ids[:nval]), "train": set(ids[nval:])}
        os.makedirs(a.out, exist_ok=True)
        for name, keep in parts.items():
            sub = {"images": [i for i in d["images"] if i["id"] in keep],
                   "annotations": [x for x in d["annotations"] if x["image_id"] in keep],
                   "categories": d["categories"]}
            q = osp.join(a.out, f"instances_{name}.json")
            json.dump(sub, open(q, "w"))
            print(f"  {name:<6} {len(sub['images']):>6,} images {len(sub['annotations']):>8,} boxes"
                  f"  -> {q}")
        # split by IMAGE, never by annotation: a detection ground truth must carry every box on
        # the image it belongs to.
        return 0

    kind, info = detect(a.root)
    print(f"format      : {kind}")
    print(f"images      : {len(info['images']):,}")
    print(f"label files : {len(info['files']):,}")

    if kind == "unknown":
        print(f"  other files: {info.get('counts')}")
        raise SystemExit(
            "Cannot identify the annotation format. Supported: COCO json, YOLO .txt, Pascal VOC "
            ".xml, LabelMe .json. Ask the user what tool produced the labels.")
    if kind == "coco":
        print(f"already COCO: {info['files'][0]}")
        print("no conversion needed -- validate it with check_dataset.py")
        return 0
    if not a.convert:
        print(f"\nrun again with --convert --out <dir> to write COCO json")
        return 0
    if not a.out:
        raise SystemExit("--convert needs --out")

    names, recs = {"yolo": from_yolo, "voc": from_voc, "labelme": from_labelme}[kind](
        a.root, info["files"], info["images"])
    if not recs:
        raise SystemExit("found label files but none matched an image by filename -- check that "
                         "labels and images share stems")
    nb = sum(len(b) for _, _, _, b in recs)
    print(f"\nparsed      : {len(recs):,} images, {nb:,} boxes, {len(names)} classes")
    print("classes     :", ", ".join(names[:12]) + (" ..." if len(names) > 12 else ""))
    empty = sum(1 for r in recs if not r[3])
    if empty:
        print(f"  {empty:,} images have no boxes")

    written = write_coco(recs, names, a.root, a.out, a.val_frac, a.seed)
    print()
    for k, (p, ni, na) in written.items():
        print(f"  {k:<6} {ni:>7,} images {na:>9,} boxes  -> {p}")
    print(f"\n  data_root is {a.root}   (file_name is relative to it)")
    print("  the vocabulary is NOT written here -- check_dataset.py --write-vocab is its one source")
    print("  next: check_dataset.py on these files, then make_preview_nb.py to see the boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
