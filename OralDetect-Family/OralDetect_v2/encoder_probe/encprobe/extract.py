"""One vector per ground-truth box: RoIAlign on the frozen encoder's last feature map."""
from __future__ import annotations

import json

import torch
from torchvision.ops import roi_align

from .data import CocoBoxes, TeethDataset, collate, condition_classes

VIEWS = ("Center", "Upper", "Down", "Left", "Right")


@torch.no_grad()
def extract(model, ds, size: int, roi: dict, batch: int, workers: int, amp: bool, device) -> dict:
    """{feat (N, C) float16, geom (N, 4) cx cy w h in [0, 1], cls (N,), group (N,)}.
    `group` is the image's fourth dataset field: an int, or a `<patient>_<view>` stem (-> view index)."""
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, collate_fn=collate)
    out = dict(feat=[], geom=[], cls=[], group=[])
    for x, boxes, cls, groups in dl:
        with torch.autocast("cuda", enabled=amp):
            fmap = model(x.to(device)).float()
        pooled = roi_align(fmap, [b.to(device) for b in boxes], roi["output_size"], spatial_scale=fmap.shape[-1] / size,
                           sampling_ratio=roi["sampling_ratio"], aligned=True).mean((2, 3))
        out["feat"].append(pooled.half().cpu())
        for b, c, g in zip(boxes, cls, groups):
            b = b / size
            out["geom"].append(torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1))
            out["cls"].append(c)
            out["group"].append(torch.full((len(c),), VIEWS.index(g.split("_")[1]) if isinstance(g, str) else g))
    return {k: torch.cat(v) for k, v in out.items()}


def extract_all(model, probe: dict, size: int, device="cuda") -> dict:
    """Every split of a probe config. Tooth probes (`yolo_root`) name the group `view`; condition
    probes (`coco`) name it `mod` and also return `classes` = {modality: class names}."""
    e, kw = probe["extract"], {}
    run = lambda ds: extract(model, ds, size, probe["roi"], e["batch"], e["workers"], e["amp"], device)
    if "coco" in probe:
        coco = {s: json.load(open(v["ann"])) for s, v in probe["coco"].items()}
        kw["classes"] = condition_classes(coco["train"], coco["test"], probe["exclude"])
        data = {s: run(CocoBoxes(probe["coco"][s]["images"], coco[s], size, kw["classes"])) for s in coco}
        key = "mod"
    else:
        data, key = {s: run(TeethDataset(probe["yolo_root"], s, size)) for s in ("train", "val", "test")}, "view"
    for d in data.values():
        d[key] = d.pop("group")
    return dict(**data, **kw)
