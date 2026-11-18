"""FDI (ISO 3950) permanent-tooth notation: the label space of this module.

A code is two digits ``QP``: quadrant Q (1 upper-right, 2 upper-left, 3 lower-left, 4 lower-right,
all from the PATIENT's point of view) and position P (1 central incisor ... 8 third molar).
Everything that needs to know "what is a class" imports it from here -- nothing else hardcodes it.
"""
from __future__ import annotations

QUADRANTS = (1, 2, 3, 4)
POSITIONS = tuple(range(1, 9))
FDI_CODES: tuple[int, ...] = tuple(q * 10 + p for q in QUADRANTS for p in POSITIONS)   # 32 classes
CODE2IDX = {c: i for i, c in enumerate(FDI_CODES)}
IDX2CODE = dict(enumerate(FDI_CODES))
CLASS_NAMES = [str(c) for c in FDI_CODES]

_MIRROR_Q = {1: 2, 2: 1, 3: 4, 4: 3}


def quadrant(code: int) -> int:
    return code // 10


def position(code: int) -> int:
    return code % 10


def arch(code: int) -> str:
    return "upper" if quadrant(code) in (1, 2) else "lower"


def mirror(code: int) -> int:
    """The same tooth on the other side of the midline (16 <-> 26). What a horizontal flip does."""
    return _MIRROR_Q[quadrant(code)] * 10 + position(code)


# Error taxonomy, ordered: the first rule that fires names the error.
RELATIONS = ("correct", "adjacent", "same_quadrant", "mirror", "same_arch", "other_arch")


def relation(gt: int, pred: int) -> str:
    """How a predicted code relates to the true one.

    adjacent       same quadrant, one position off (15 -> 14), or the two central incisors (11 -> 21)
    same_quadrant  same quadrant, >= 2 positions off
    mirror         right tooth, wrong side (16 -> 26)
    same_arch      other side of the same arch, different position
    other_arch     upper <-> lower
    """
    if gt == pred:
        return "correct"
    if quadrant(gt) == quadrant(pred):
        return "adjacent" if abs(position(gt) - position(pred)) == 1 else "same_quadrant"
    if arch(gt) != arch(pred):
        return "other_arch"
    if position(gt) == 1 and position(pred) == 1:
        return "adjacent"                      # 11 <-> 21, 31 <-> 41: neighbours across the midline
    return "mirror" if mirror(gt) == pred else "same_arch"


# Tooth type <-> position digit, for datasets that label the type but not the quadrant.
POSITION_NAMES = {1: "central incisor", 2: "lateral incisor", 3: "canine", 4: "first premolar",
                  5: "second premolar", 6: "first molar", 7: "second molar", 8: "third molar"}
NAME2POSITION = {n: p for p, n in POSITION_NAMES.items()}
