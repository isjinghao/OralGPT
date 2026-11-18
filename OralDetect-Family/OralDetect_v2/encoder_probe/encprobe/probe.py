"""Train a classifier on cached tooth features and score it.

Three readouts of the same cache:
    linear        logistic regression on the feature alone           (what the encoder already separates)
    mlp           one hidden layer on the feature alone               (what it holds non-linearly)
    mlp+geom      one hidden layer on feature + box position/size     (feature value on top of geometry)
and one floor shared by all encoders:
    geom          one hidden layer on box position/size + view only
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import FDI_CODES

CODES = torch.tensor(FDI_CODES)
RIGHT = torch.tensor([1, 4])           # quadrants on the patient's right


def _inputs(d: dict, mode: str, stats=None):
    geom = torch.cat([d["geom"], F.one_hot(d["view"], 5).float()], 1)
    x = geom if mode == "geom" else d["feat"].float()
    stats = stats or (x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-6))
    x = (x - stats[0]) / stats[1]
    return (torch.cat([x, geom], 1) if mode == "mlp+geom" else x), stats


def _score(logits: torch.Tensor, y: torch.Tensor, view: torch.Tensor) -> dict:
    pred, gt = CODES[logits.argmax(1).cpu()], CODES[y.cpu()]
    acc = lambda m: round(float(m.float().mean()), 4)
    out = dict(fdi=acc(pred == gt), position=acc(pred % 10 == gt % 10), quadrant=acc(pred // 10 == gt // 10),
               arch=acc((pred // 10 <= 2) == (gt // 10 <= 2)),
               side=acc(torch.isin(pred // 10, RIGHT) == torch.isin(gt // 10, RIGHT)))
    for i, v in enumerate(("Center", "Upper", "Down", "Left", "Right")):
        m = view.cpu() == i
        out[f"fdi_{v}"] = acc((pred == gt)[m])
    return out


def run_probe(train: dict, val: dict, test: dict, mode: str, hp: dict, device) -> dict:
    torch.manual_seed(hp["seed"])
    xtr, stats = _inputs(train, mode)
    xva, xte = _inputs(val, mode, stats)[0].to(device), _inputs(test, mode, stats)[0].to(device)
    xtr, ytr = xtr.to(device), train["cls"].to(device)
    d = xtr.shape[1]
    net = nn.Linear(d, 32) if mode == "linear" else nn.Sequential(nn.Linear(d, hp["mlp_hidden"]), nn.GELU(), nn.Linear(hp["mlp_hidden"], 32))
    net.to(device)
    steps = hp["epochs"] * -(-len(xtr) // hp["batch"])
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp["lr"], total_steps=steps)
    best = (-1.0, None, -1)
    for ep in range(hp["epochs"]):
        net.train()
        for idx in torch.randperm(len(xtr), device=device).split(hp["batch"]):
            opt.zero_grad()
            F.cross_entropy(net(xtr[idx]), ytr[idx]).backward()
            opt.step(), sched.step()
        net.eval()
        with torch.no_grad():
            va = float((net(xva).argmax(1).cpu() == val["cls"]).float().mean())
            if va > best[0]:                                   # model selection on val, report test
                best = (va, _score(net(xte), test["cls"], test["view"]), ep + 1)
    return dict(mode=mode, val_fdi=round(best[0], 4), best_epoch=best[2], test=best[1], n_test=len(test["cls"]))
