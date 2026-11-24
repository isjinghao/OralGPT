"""Image sets without tooth numbers, and the COCO file their predicted numbers are written to."""
from __future__ import annotations

from pathlib import Path

from ..fdi import CODE2IDX, FDI_CODES

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(ds: dict) -> list[dict]:
    """[{stem, path}] for one `datasets` entry: files in `dirs` whose name starts with `prefix`.
    A file name met in a second directory is skipped, so the result has one record per name."""
    out, seen = [], set()
    for d in ds["dirs"]:
        for p in sorted(Path(d).iterdir()):
            if p.suffix.lower() in IMAGE_EXT and p.name.startswith(ds.get("prefix", "")) and p.name not in seen:
                seen.add(p.name)
                out.append(dict(stem=p.stem, path=str(p)))
    stems = [r["stem"] for r in out]
    if len(set(stems)) != len(stems):
        raise ValueError(f"{ds['name']}: two images share a stem with different extensions")
    return out


def to_coco(preds: list[dict], names: dict[str, str]) -> dict:
    """Predictions of engine.predict -> COCO dict. Category ids are 1..32 in FDI order, named by the code;
    each annotation keeps the model `score`. `names` maps stem -> file name on disk."""
    images, anns = [], []
    for i, p in enumerate(preds, 1):
        images.append(dict(id=i, file_name=names[p["stem"]], width=p["size"][0], height=p["size"][1]))
        for box, code, score in zip(p["boxes"], p["codes"], p["scores"]):
            x1, y1, x2, y2 = box
            anns.append(dict(id=len(anns) + 1, image_id=i, category_id=CODE2IDX[code] + 1, fdi=code, score=score,
                             bbox=[x1, y1, round(x2 - x1, 1), round(y2 - y1, 1)], area=round((x2 - x1) * (y2 - y1), 1), iscrowd=0))
    return dict(images=images, annotations=anns, categories=[dict(id=i + 1, name=str(c)) for i, c in enumerate(FDI_CODES)])


def image_stats(p: dict) -> dict:
    """Per-image signals used to rank and summarise pseudo-labels."""
    n = len(p["codes"])
    return dict(n=n, mean_score=sum(p["scores"]) / n if n else 0.0, n_repeat=n - len(set(p["codes"])))
