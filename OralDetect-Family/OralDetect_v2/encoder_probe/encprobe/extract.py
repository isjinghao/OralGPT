"""One vector per ground-truth tooth: RoIAlign on the frozen encoder's last feature map."""
from __future__ import annotations

import torch
from torchvision.ops import roi_align

from .data import TeethDataset, collate

VIEWS = ("Center", "Upper", "Down", "Left", "Right")


@torch.no_grad()
def extract_split(model, yolo_root, split: str, size: int, roi: dict, batch: int, workers: int, amp: bool, device) -> dict:
    """{feat (N, C) float16, geom (N, 4) cx cy w h in [0, 1], cls (N,), view (N,)}"""
    dl = torch.utils.data.DataLoader(TeethDataset(yolo_root, split, size), batch_size=batch, num_workers=workers,
                                     collate_fn=collate)
    out = dict(feat=[], geom=[], cls=[], view=[])
    for x, boxes, cls, stems in dl:
        with torch.autocast("cuda", enabled=amp):
            fmap = model(x.to(device)).float()
        pooled = roi_align(fmap, [b.to(device) for b in boxes], roi["output_size"], spatial_scale=fmap.shape[-1] / size,
                           sampling_ratio=roi["sampling_ratio"], aligned=True).mean((2, 3))
        out["feat"].append(pooled.half().cpu())
        for b, c, s in zip(boxes, cls, stems):
            b = b / size
            out["geom"].append(torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1))
            out["cls"].append(c)
            out["view"].append(torch.full((len(c),), VIEWS.index(s.split("_")[1])))
    return {k: torch.cat(v) for k, v in out.items()}
