"""Reader for the raw IOP-Compass release.

Layout:  <raw_root>/Patient_<id>/IOP_<View>_<id>.{png,json}     View in VIEWS
JSON:    {"teeth": [{"FDI_NUM": "16", "contours": [[[x, y], ...], ...], "x_min": ..., ...}, ...]}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..fdi import CODE2IDX

VIEWS = ("Center", "Upper", "Down", "Left", "Right")


@dataclass(frozen=True)
class Record:
    patient: str            # "499"
    view: str               # one of VIEWS
    image: Path
    ann: Path

    @property
    def stem(self) -> str:  # file stem used everywhere downstream; view is recoverable from it
        return f"P{int(self.patient):04d}_{self.view}"


@dataclass(frozen=True)
class Tooth:
    code: int                           # FDI code
    polygon: list[tuple[float, float]]  # pixel coords in the ORIGINAL image


def list_records(raw_root: str | Path) -> list[Record]:
    recs = []
    for pdir in sorted(Path(raw_root).glob("Patient_*"), key=lambda p: int(p.name.split("_")[1])):
        pid = pdir.name.split("_")[1]
        for v in VIEWS:
            img, ann = pdir / f"IOP_{v}_{pid}.png", pdir / f"IOP_{v}_{pid}.json"
            if not (img.exists() and ann.exists()):
                raise FileNotFoundError(f"incomplete pair: {img}")
            recs.append(Record(pid, v, img, ann))
    return recs


def load_teeth(ann: str | Path) -> list[Tooth]:
    """One Tooth per annotated instance. A tooth split into several contours (33 of 88,601) keeps
    its largest one -- YOLO-seg labels are single polygons."""
    teeth = []
    for t in json.load(open(ann))["teeth"]:
        code = int(t["FDI_NUM"])
        if code not in CODE2IDX:
            raise ValueError(f"{ann}: FDI code {code} outside the permanent dentition")
        contours = [c for c in t["contours"] if len(c) >= 3]
        if not contours:
            continue
        teeth.append(Tooth(code, [tuple(p) for p in max(contours, key=_area)]))
    return teeth


def _area(poly) -> float:
    s = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2
