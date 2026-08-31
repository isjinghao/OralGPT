#!/usr/bin/env python
"""Validate a COCO dataset against what OralDetect's loader actually requires, and emit the
vocabulary files if they are missing.

    python check_dataset.py --train <json> [--val <json>] [--test <json>] \
                            --images <dir> [--out <dir>] [--write-vocab]

Every check here exists because the failure it catches is SILENT. The loader does not raise on a
mismatched category name, an unreadable image, or an out-of-range box -- it drops the annotation
and trains on what is left, so the only symptom is a number that comes out low days later.
"""
import argparse
import json
import os
import os.path as osp
import sys
from collections import Counter

OK, WARN, BAD = "\033[32mOK\033[0m", "\033[33mWARN\033[0m", "\033[31mFAIL\033[0m"


def load(p):
    if not osp.isfile(p):
        sys.exit(f"FAIL: no such file -- {p}")
    with open(p) as f:
        d = json.load(f)
    for k in ("images", "annotations", "categories"):
        if k not in d:
            sys.exit(f"FAIL: {p} has no `{k}` -- not a COCO detection file")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val")
    ap.add_argument("--test")
    ap.add_argument("--images", required=True, help="data_root; file_name is relative to it")
    ap.add_argument("--out", default=".", help="where to write class_names/class_texts json")
    ap.add_argument("--write-vocab", action="store_true")
    ap.add_argument("--sample", type=int, default=300, help="images to stat-check on disk")
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


    splits = {"train": a.train}
    if a.val:
        splits["val"] = a.val
    if a.test:
        splits["test"] = a.test
    data = {k: load(v) for k, v in splits.items()}
    fails = []

    print("=" * 74)
    for k, d in data.items():
        nb = len(d["annotations"])
        print(f"  {k:<6} {len(d['images']):>7,} images  {nb:>9,} boxes  "
              f"{len(d['categories']):>4} categories")

    # ---- 1. the vocabulary, taken from train ------------------------------------------------
    cats = sorted(data["train"]["categories"], key=lambda c: c["id"])
    names = [c["name"] for c in cats]
    print("=" * 74)
    if len(names) != len(set(names)):
        dup = [n for n, c in Counter(names).items() if c > 1]
        print(f"  [{BAD}] duplicate category names: {dup}")
        fails.append("duplicate category names")
    else:
        print(f"  [{OK}] {len(names)} distinct category names")

    # THE trap: the loader resolves categories BY NAME against class_names, and the label index is
    # the position in that list. A name present in one split but not another silently loses boxes.
    for k, d in data.items():
        if k == "train":
            continue
        other = {c["name"] for c in d["categories"]}
        missing = other - set(names)
        if missing:
            print(f"  [{BAD}] {k} has category names absent from train: {sorted(missing)[:6]}")
            print("         -> their annotations would be DROPPED without any error")
            fails.append(f"{k} category names not in train")
        else:
            print(f"  [{OK}] {k} category names are a subset of train's")

    if len(names) > 256:
        print(f"  [{BAD}] {len(names)} classes > 256")
        print("         -> the head sizes its classifier as max(256, num_classes); past 256 the")
        print("            cls_preds convs change shape and the checkpoint's are dropped.")
        fails.append("more than 256 classes")
    else:
        print(f"  [{OK}] {len(names)} classes <= 256 (shape-safe for the released checkpoint)")

    # ---- 1b. categories that carry no boxes ---------------------------------------------------
    box_per_cat = Counter()
    for k, d in data.items():
        id2name = {c["id"]: c["name"] for c in d["categories"]}
        for x in d["annotations"]:
            box_per_cat[id2name.get(x["category_id"], "??")] += 1
    unused = [n for n in names if box_per_cat[n] == 0]
    thin = [(n, box_per_cat[n]) for n in names if 0 < box_per_cat[n] < 10]
    if unused:
        print(f"  [{WARN}] {len(unused)}/{len(names)} categories have NO boxes in any split")
        print(f"         {sorted(unused)[:8]}{' ...' if len(unused) > 8 else ''}")
        print("         -> declared in `categories` but never annotated. They will train on nothing.")
        print("            Drop them from class_names/class_texts unless you want the model to keep")
        print("            recognising them (it can: the text tower carries them zero-shot).")
    else:
        print(f"  [{OK}] every category has at least one box")
    if thin:
        print(f"  [{WARN}] {len(thin)} categories have <10 boxes: "
              f"{[f'{n}({c})' for n, c in thin[:6]]}")
        print("         -> too few to learn; merge them or accept near-zero AP.")

    # ---- 2. annotations point at real images, boxes are sane -------------------------------
    for k, d in data.items():
        ids = {im["id"] for im in d["images"]}
        orphan = sum(1 for x in d["annotations"] if x["image_id"] not in ids)
        catids = {c["id"] for c in d["categories"]}
        badcat = sum(1 for x in d["annotations"] if x["category_id"] not in catids)
        degen = sum(1 for x in d["annotations"]
                    if len(x.get("bbox", [])) != 4 or x["bbox"][2] <= 0 or x["bbox"][3] <= 0)
        empty = sum(1 for i in ids if i not in {x["image_id"] for x in d["annotations"]})
        tag = BAD if (orphan or badcat) else (WARN if degen else OK)
        print(f"  [{tag}] {k:<6} orphan-ann {orphan} · bad category_id {badcat} · "
              f"degenerate boxes {degen} · images with 0 boxes {empty}")
        if orphan or badcat:
            fails.append(f"{k}: {orphan} orphan / {badcat} bad-category annotations")

    # ---- 3. images resolve on disk ----------------------------------------------------------
    if not osp.isdir(a.images):
        print(f"  [{BAD}] --images is not a directory: {a.images}")
        fails.append("images dir missing")
    else:
        for k, d in data.items():
            fns = [im["file_name"] for im in d["images"]]
            probe = fns[:: max(1, len(fns) // a.sample)][:a.sample]
            miss = [f for f in probe if not osp.isfile(osp.join(a.images, f))]
            if miss:
                print(f"  [{BAD}] {k}: {len(miss)}/{len(probe)} sampled images not under "
                      f"{a.images}")
                print(f"         e.g. {miss[:3]}")
                fails.append(f"{k}: images do not resolve")
            else:
                print(f"  [{OK}] {k}: {len(probe)} sampled file_names all resolve")

    # ---- 4. the modality key decides which evaluator is legal --------------------------------
    has_mod = all("modality" in im for im in data["train"]["images"])
    print(f"  [{OK}] evaluator: {'per_modality is available' if has_mod else 'coco'}"
          f"{'' if has_mod else '  (no `modality` key on images -- per_modality would raise)'}")

    # ---- 5. vocabulary files ------------------------------------------------------------------
    if a.write_vocab:
        os.makedirs(a.out, exist_ok=True)
        pn, pt = osp.join(a.out, "class_names.json"), osp.join(a.out, "class_texts.json")
        json.dump(names, open(pn, "w"), indent=1, ensure_ascii=False)
        # one prompt per class, the name itself. Edit these -- the text tower embeds them and that
        # embedding IS the class prototype, so clinical wording beats a dataset code.
        json.dump([[n] for n in names], open(pt, "w"), indent=1, ensure_ascii=False)
        print("=" * 74)
        print(f"  wrote {pn}\n  wrote {pt}   ({len(names)} classes, in categories[].id order)")
        print("  ⚠️  class_texts holds the raw names. Rewrite anything that is a code or an")
        print("      abbreviation into what a clinician would write -- that string is the classifier.")

    print("=" * 74)
    if fails:
        print(f"  {len(fails)} blocking problem(s):")
        for f in fails:
            print(f"    - {f}")
        return 1
    print("  dataset is usable as-is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
