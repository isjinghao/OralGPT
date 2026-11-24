"""IOP-Compass in the YOLO layout written by tooth_id: square-letterboxed images + tooth boxes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

FDI_CODES = tuple(q * 10 + p for q in (1, 2, 3, 4) for p in range(1, 9))     # class index -> FDI code


class TeethDataset(torch.utils.data.Dataset):
    def __init__(self, yolo_root: str | Path, split: str, size: int):
        self.root, self.split, self.size = Path(yolo_root), split, size
        self.stems = sorted(p.stem for p in (self.root / "images" / split).glob("*.jpg"))

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem, S = self.stems[i], self.size
        im = Image.open(self.root / "images" / self.split / f"{stem}.jpg").convert("RGB")
        w, h = im.size
        s = S / max(w, h)
        nw, nh = round(w * s), round(h * s)
        canvas = Image.new("RGB", (S, S))                       # top-left anchored, zero padded
        canvas.paste(im.resize((nw, nh), Image.BICUBIC), (0, 0))
        x = torch.from_numpy(np.asarray(canvas)).permute(2, 0, 1).float() / 255
        cls, boxes = [], []
        for line in (self.root / "labels" / self.split / f"{stem}.txt").read_text().splitlines():
            v = line.split()
            if len(v) < 7:
                continue
            xs, ys = [float(a) * nw for a in v[1::2]], [float(a) * nh for a in v[2::2]]
            cls.append(int(v[0]))
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
        return x, torch.tensor(boxes).reshape(-1, 4), torch.tensor(cls, dtype=torch.long), stem


def collate(batch):
    xs, boxes, cls, stems = zip(*batch)
    return torch.stack(xs), list(boxes), list(cls), list(stems)


def condition_classes(train: dict, test: dict, exclude: list[str]) -> dict[str, list[str]]:
    """{modality: sorted class names} for the condition probe: classes not matched by an `exclude`
    regex that have boxes in both COCO dicts in that modality; modalities with one class are dropped."""
    import re

    def seen(d):
        mod = {i["id"]: i["modality"] for i in d["images"]}
        name = {c["id"]: c["name"] for c in d["categories"]}
        return {(mod[a["image_id"]], name[a["category_id"]]) for a in d["annotations"]}

    out: dict[str, list[str]] = {}
    for m, n in sorted(seen(train) & seen(test)):
        if not any(re.search(p, n) for p in exclude):
            out.setdefault(m, []).append(n)
    return {m: v for m, v in out.items() if len(v) > 1}


class CocoBoxes(torch.utils.data.Dataset):
    """Images of a COCO dict that hold at least one box of `classes`; the label of a box is its index
    in its modality's class list, the group of an image is its modality's index."""

    def __init__(self, images_root: str | Path, coco: dict, size: int, classes: dict[str, list[str]]):
        self.root, self.size, self.mods = Path(images_root), size, sorted(classes)
        name = {c["id"]: c["name"] for c in coco["categories"]}
        mod = {i["id"]: i["modality"] for i in coco["images"]}
        per: dict[int, list] = {}
        for a in coco["annotations"]:
            m, n, (x, y, w, h) = mod[a["image_id"]], name[a["category_id"]], a["bbox"]
            if n in classes.get(m, ()) and w > 1 and h > 1:
                per.setdefault(a["image_id"], []).append((x, y, x + w, y + h, classes[m].index(n)))
        self.items = [(i["file_name"], self.mods.index(i["modality"]), per[i["id"]]) for i in coco["images"] if i["id"] in per]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        file, group, rows, S = *self.items[i], self.size
        im = Image.open(self.root / file).convert("RGB")
        s = S / max(im.size)
        canvas = Image.new("RGB", (S, S))                       # top-left anchored, zero padded
        canvas.paste(im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC), (0, 0))
        x = torch.from_numpy(np.asarray(canvas)).permute(2, 0, 1).float() / 255
        rows = torch.tensor(rows, dtype=torch.float32)
        return x, rows[:, :4] * s, rows[:, 4].long(), group
