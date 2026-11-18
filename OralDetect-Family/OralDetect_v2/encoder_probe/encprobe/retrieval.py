"""Image-to-image retrieval with frozen global features."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class BenchImages(torch.utils.data.Dataset):
    def __init__(self, bench_root: str | Path, size: int):
        self.root, self.size = Path(bench_root), size
        self.rows = [json.loads(l) for l in open(self.root / "test_clip.jsonl")]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        im = Image.open(self.root / "images" / self.rows[i]["image"]).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255


@torch.no_grad()
def global_features(model, ds, batch: int, workers: int, amp: bool, device="cuda") -> torch.Tensor:
    out = []
    for x in torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers):
        with torch.autocast("cuda", enabled=amp):
            out.append(model(x.to(device)).float().mean((2, 3)).cpu())
    return torch.cat(out)


def retrieval_scores(feat: torch.Tensor, rows: list[dict], ks=(1, 5)) -> dict:
    """Per modality: gallery = the other images of that modality; relevant = same category.
    Queries whose category has no other image are skipped. Returns R@k (any hit in top k), mAP, and
    `chance` = the R@1 of a random ranking."""
    feat, out = F.normalize(feat - feat.mean(0, keepdim=True), dim=1), {}
    for mod in sorted({r["modality"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["modality"] == mod]
        cats = [rows[i]["category"] for i in idx]
        ids = torch.tensor([sorted(set(cats)).index(c) for c in cats])
        sim = feat[idx] @ feat[idx].T
        sim.fill_diagonal_(-2)
        rel = (ids[:, None] == ids[None]).float()
        rel.fill_diagonal_(0)
        keep = rel.sum(1) > 0
        ranked = rel.gather(1, sim.argsort(1, descending=True))[keep]            # relevance in rank order
        prec = ranked.cumsum(1) / torch.arange(1, ranked.shape[1] + 1)
        res = {f"R@{k}": round(float((ranked[:, :k].sum(1) > 0).float().mean()), 4) for k in ks}
        res["mAP"] = round(float(((prec * ranked).sum(1) / ranked.sum(1)).mean()), 4)
        res["chance"] = round(float((rel.sum(1)[keep] / (len(idx) - 1)).mean()), 4)
        res.update(n_queries=int(keep.sum()), n_categories=len(set(cats)), largest_category=Counter(cats).most_common(1)[0][1])
        out[mod] = res
    out["macro"] = {m: round(float(np.mean([v[m] for k, v in out.items()])), 4) for m in [f"R@{k}" for k in ks] + ["mAP", "chance"]}
    return out
