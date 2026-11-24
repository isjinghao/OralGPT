"""End-to-end fine-tuning of an encoder on the tooth-FDI task, on a subset of training patients."""
from __future__ import annotations

import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from .data import TeethDataset, collate
from .extract import VIEWS
from .probe import _score


def patient_subset(stems: list[str], fraction: float, seed: int) -> list[str]:
    """Stems of the first `fraction` of patients in a seeded order: smaller fractions are subsets of larger ones."""
    patients = sorted({s.split("_")[0] for s in stems})
    random.Random(seed).shuffle(patients)
    keep = set(patients[: max(1, round(fraction * len(patients)))])
    return [s for s in stems if s.split("_")[0] in keep]


class ToothClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, dim: int, roi: dict, size: int, hidden: int):
        super().__init__()
        self.encoder, self.roi, self.size = encoder, roi, size
        self.head = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 32))

    def forward(self, x, boxes):
        fmap = self.encoder(x)
        pooled = roi_align(fmap.float(), boxes, self.roi["output_size"], spatial_scale=fmap.shape[-1] / self.size,
                           sampling_ratio=self.roi["sampling_ratio"], aligned=True).mean((2, 3))
        return self.head(pooled)


@torch.no_grad()
def evaluate(model, loader, device, amp: bool) -> dict:
    model.eval()
    logits, ys, views = [], [], []
    for x, boxes, cls, stems in loader:
        with torch.autocast("cuda", enabled=amp):
            logits.append(model(x.to(device), [b.to(device) for b in boxes]).float().cpu())
        ys += cls
        views += [torch.full((len(c),), VIEWS.index(s.split("_")[1])) for c, s in zip(cls, stems)]
    return _score(torch.cat(logits), torch.cat(ys), torch.cat(views))


def finetune(encoder, dim: int, cfg: dict, fraction: float, device="cuda", save_path=None) -> dict:
    """`save_path`: where the encoder weights of the best validation step are written (optional)."""
    t, size = cfg["train"], cfg["image_size"]
    torch.manual_seed(t["seed"])
    train_ds = TeethDataset(cfg["yolo_root"], "train", size)
    train_ds.stems = patient_subset(train_ds.stems, fraction, cfg["subset_seed"])
    mk = lambda ds, shuffle: torch.utils.data.DataLoader(ds, batch_size=t["batch"], shuffle=shuffle, drop_last=shuffle,
                                                         num_workers=t["workers"], collate_fn=collate, persistent_workers=shuffle)
    train_dl = mk(train_ds, True)
    val_dl, test_dl = mk(TeethDataset(cfg["yolo_root"], "val", size), False), mk(TeethDataset(cfg["yolo_root"], "test", size), False)

    model = ToothClassifier(encoder.requires_grad_(True), dim, cfg["roi"], size, t["mlp_hidden"]).to(device)
    opt = torch.optim.AdamW([dict(params=model.encoder.parameters(), lr=t["lr_backbone"]),
                             dict(params=model.head.parameters(), lr=t["lr_head"])], weight_decay=t["weight_decay"])
    ramp = lambda s: min(1.0, (s + 1) / t["warmup_steps"]) * 0.5 * (1 + math.cos(math.pi * s / t["steps"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, ramp)
    scaler = torch.amp.GradScaler("cuda", enabled=t["amp"])

    step, t0, curve, best = 0, time.time(), [], None
    while step < t["steps"]:
        for x, boxes, cls, _ in train_dl:
            model.train()
            with torch.autocast("cuda", enabled=t["amp"]):
                loss = F.cross_entropy(model(x.to(device), [b.to(device) for b in boxes]).float(), torch.cat(cls).to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt), scaler.update(), sched.step()
            step += 1
            if step % t["eval_every"] == 0 or step == t["steps"]:
                val, test = evaluate(model, val_dl, device, t["amp"]), evaluate(model, test_dl, device, t["amp"])
                curve.append(dict(step=step, loss=round(float(loss), 4), val=val["fdi"], test=test["fdi"]))
                print(f"  step {step} loss {float(loss):.3f} val {val['fdi']} test {test['fdi']} ({time.time() - t0:.0f}s)", flush=True)
                if best is None or val["fdi"] > best["val_fdi"]:
                    best = dict(val_fdi=val["fdi"], step=step, test=test)
                    if save_path is not None:
                        torch.save({k: v.cpu() for k, v in model.encoder.body.state_dict().items()}, save_path)
            if step >= t["steps"]:
                break
    return dict(fraction=fraction, n_patients=len({s.split("_")[0] for s in train_ds.stems}), n_images=len(train_ds),
                minutes=round((time.time() - t0) / 60, 1), curve=curve, **best)
