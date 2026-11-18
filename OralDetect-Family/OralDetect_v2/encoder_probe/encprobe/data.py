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
