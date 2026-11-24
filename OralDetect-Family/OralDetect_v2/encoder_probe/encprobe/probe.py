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


def _fit(xtr, ytr, xva, yva, xte, n_cls: int, linear: bool, hp: dict, metric):
    """Train a linear layer or a one-hidden-layer MLP; return (best val metric, test logits at that epoch, epoch).
    `metric(pred, y)` scores val predictions on CPU and selects the epoch."""
    d = xtr.shape[1]
    net = nn.Linear(d, n_cls) if linear else nn.Sequential(nn.Linear(d, hp["mlp_hidden"]), nn.GELU(), nn.Linear(hp["mlp_hidden"], n_cls))
    net.to(xtr.device)
    steps = hp["epochs"] * -(-len(xtr) // hp["batch"])
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp["lr"], total_steps=steps)
    best = (-1.0, None, -1)
    for ep in range(hp["epochs"]):
        net.train()
        for idx in torch.randperm(len(xtr), device=xtr.device).split(hp["batch"]):
            opt.zero_grad()
            F.cross_entropy(net(xtr[idx]), ytr[idx]).backward()
            opt.step(), sched.step()
        net.eval()
        with torch.no_grad():
            va = metric(net(xva).argmax(1).cpu(), yva)
            if va > best[0]:                                   # model selection on val, report test
                best = (va, net(xte).cpu(), ep + 1)
    return best


def run_probe(train: dict, val: dict, test: dict, mode: str, hp: dict, device) -> dict:
    torch.manual_seed(hp["seed"])
    xtr, stats = _inputs(train, mode)
    xva, xte = _inputs(val, mode, stats)[0].to(device), _inputs(test, mode, stats)[0].to(device)
    va, logits, ep = _fit(xtr.to(device), train["cls"].to(device), xva, val["cls"], xte, 32, mode == "linear", hp,
                          lambda p, y: float((p == y).float().mean()))
    return dict(mode=mode, val_fdi=round(va, 4), best_epoch=ep, test=_score(logits, test["cls"], test["view"]), n_test=len(test["cls"]))


def _class_stats(pred: torch.Tensor, y: torch.Tensor, n_cls: int):
    """Per-class recall and F1 over the classes present in y."""
    conf = torch.zeros(n_cls, n_cls).index_put_((y, pred), torch.ones(len(y)), accumulate=True)
    tp, has = conf.diag(), conf.sum(1) > 0
    recall = tp / conf.sum(1).clamp_min(1)
    f1 = 2 * tp / (conf.sum(0) + conf.sum(1)).clamp_min(1)
    return recall, f1, has


def balanced_accuracy(pred: torch.Tensor, y: torch.Tensor, n_cls: int) -> float:
    recall, _, has = _class_stats(pred, y, n_cls)
    return float(recall[has].mean())


def run_condition_probe(train: dict, val: dict, test: dict, mod: int, names: list[str], mode: str, hp: dict, device) -> dict:
    """Classify the boxes of one modality into its condition classes. mode: linear | mlp (feature
    alone) | geom (box position and size alone, the floor). Epoch selected on val balanced accuracy."""
    torch.manual_seed(hp["seed"])
    pick = lambda d: (d["geom"] if mode == "geom" else d["feat"].float())[d["mod"] == mod]
    xtr, n = pick(train), len(names)
    mean, std = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-6)
    x = {k: ((pick(d) - mean) / std).to(device) for k, d in (("train", train), ("val", val), ("test", test))}
    y = {k: d["cls"][d["mod"] == mod] for k, d in (("train", train), ("val", val), ("test", test))}
    va, logits, ep = _fit(x["train"], y["train"].to(device), x["val"], y["val"], x["test"], n, mode == "linear", hp,
                          lambda p, t: balanced_accuracy(p, t, n))
    pred = logits.argmax(1)
    recall, f1, has = _class_stats(pred, y["test"], n)
    r = lambda v: round(float(v), 4)
    return dict(mode=mode, val_balanced=r(va), best_epoch=ep, n_train=len(y["train"]), n_test=len(y["test"]),
                test=dict(balanced=r(recall[has].mean()), macro_f1=r(f1[has].mean()), acc=r((pred == y["test"]).float().mean()),
                          recall={names[i]: r(recall[i]) for i in range(n) if has[i]}))
