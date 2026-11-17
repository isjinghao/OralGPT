"""Pure-python checks of the label space, the split and the matcher. No GPU, no data.
    python -m pytest tests -q        (or: python tests/test_core.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toothid.data.split import split_patients
from toothid.eval.id_metrics import IDAccumulator
from toothid.fdi import FDI_CODES, mirror, relation


def test_label_space():
    assert len(FDI_CODES) == 32 and FDI_CODES[0] == 11 and FDI_CODES[-1] == 48
    assert mirror(16) == 26 and mirror(41) == 31 and all(mirror(mirror(c)) == c for c in FDI_CODES)


def test_relation():
    assert relation(16, 16) == "correct"
    assert relation(15, 14) == "adjacent" and relation(11, 21) == "adjacent"
    assert relation(13, 16) == "same_quadrant"
    assert relation(16, 26) == "mirror" and relation(16, 25) == "same_arch"
    assert relation(16, 46) == "other_arch"


def test_split_is_patient_level_and_deterministic():
    ids = [str(i) for i in range(1, 1001)]
    r = {"train": 0.8, "val": 0.1, "test": 0.1}
    a, b = split_patients(ids, r, 0), split_patients(ids[::-1], r, 0)
    assert a == b and len(a) == 1000
    assert [sum(v == s for v in a.values()) for s in r] == [800, 100, 100]


def test_id_metrics():
    acc = IDAccumulator(0.5)
    gt_boxes = [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]]
    acc.add("Center", [11, 12, 13], gt_boxes, [11, 13, 21], [[0, 0, 10, 10], [20, 0, 30, 10], [100, 100, 110, 110]])
    o = acc.summary()["overall"]
    assert (o["n_found"], o["n_correct"]) == (2, 1)
    assert o["det_recall"] == round(2 / 3, 4) and o["id_accuracy"] == 0.5 and o["e2e_recall"] == round(1 / 3, 4)
    assert acc.summary()["relations"]["adjacent"]["n"] == 1


def test_configs_are_relative_and_resolve_under_roots():
    from toothid.config import MODULE_ROOT, load_data_cfg, load_model_cfg, load_yaml, run_dir
    roots = {"data_root": "/D", "exp_root": "/E"}
    for f in sorted((MODULE_ROOT / "configs").rglob("*.yaml")):
        assert "/ix/" not in f.read_text() and "/ihome/" not in f.read_text(), f
    for f in sorted((MODULE_ROOT / "configs/data").glob("*.yaml")):
        assert not load_yaml(f)["raw_root"].startswith("/")
        d = load_data_cfg(f, roots)
        assert d["raw_root"].startswith("/D/") and d["yolo_root"].startswith("/D/")
    for f in sorted((MODULE_ROOT / "configs/model").glob("*.yaml")):
        m = load_model_cfg(f, roots)
        assert m["weights"].startswith("/E/") and str(run_dir(m)).startswith("/E/")


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
