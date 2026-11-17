"""Patient-level split. The five views of a patient never straddle two splits."""
from __future__ import annotations

import random


def split_patients(patients: list[str], ratios: dict[str, float], seed: int) -> dict[str, str]:
    """patient id -> split name. Deterministic in (sorted patients, ratios, seed)."""
    if abs(sum(ratios.values()) - 1) > 1e-6:
        raise ValueError(f"split ratios must sum to 1: {ratios}")
    ids = sorted(set(patients), key=int)
    random.Random(seed).shuffle(ids)
    out, start, names = {}, 0, list(ratios)
    for i, name in enumerate(names):
        end = len(ids) if i == len(names) - 1 else start + round(ratios[name] * len(ids))
        out.update({p: name for p in ids[start:end]})
        start = end
    return out
