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


def test_unlabeled_listing_and_coco():
    import tempfile

    from toothid.data.unlabeled import image_stats, list_images, to_coco

    a, b = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    for d, names in ((a, ["X__1.jpg", "X__2.png", "Y__1.jpg", "note.txt"]), (b, ["X__1.jpg", "X__3.jpg"])):
        for n in names:
            (d / n).write_bytes(b"")
    recs = list_images(dict(name="X", prefix="X__", dirs=[str(a), str(b)]))
    assert [r["stem"] for r in recs] == ["X__1", "X__2", "X__3"] and recs[0]["path"].startswith(str(a))
    p = dict(stem="X__1", size=[100, 80], boxes=[[10, 20, 30, 50], [40, 20, 60, 50]], codes=[11, 11], scores=[0.9, 0.5])
    c = to_coco([p], {"X__1": "X__1.jpg"})
    assert len(c["categories"]) == 32 and c["categories"][0] == dict(id=1, name="11")
    assert c["annotations"][0]["bbox"] == [10, 20, 20, 30] and c["annotations"][1]["fdi"] == 11
    s = image_stats(p)
    assert s["n"] == 2 and s["n_repeat"] == 1 and abs(s["mean_score"] - 0.7) < 1e-9
    assert image_stats(dict(codes=[], scores=[]))["mean_score"] == 0.0


def test_typed_union_codes():
    from toothid.data.typed_union import box_label_line, complete_codes

    gt = [[0, 0, 10, 10], [12, 0, 22, 10], [24, 0, 34, 10], [36, 0, 46, 10]]        # 12 11 | 21 22, seen from the front
    pos = [2, 1, 1, 2]
    pred = [[0, 0, 10, 10], [12, 0, 22, 10], [24, 0, 34, 10]]
    assert complete_codes(pos, gt, [12, 11, 21], pred, 0.5) == [12, 11, 21, 22]      # 4th tooth: nearest neighbour is 21
    assert complete_codes(pos, gt, [12, 11, 11], pred, 0.5) is None                  # two teeth would both be 11
    assert complete_codes([6], [[0, 0, 10, 10]], [], [], 0.5) is None                # nothing to borrow a quadrant from
    assert box_label_line(5, [10, 20, 30, 40], 100, 80) == "5 0.100000 0.250000 0.300000 0.250000 0.300000 0.500000 0.100000 0.500000"


def test_anterior12_and_borrowed_teeth():
    from toothid.config import MODULE_ROOT, load_union_cfg
    from toothid.data.typed_union import ANTERIOR_LOWER, ANTERIOR_UPPER, anterior12_codes, other_teeth

    upper = [[x, 0, x + 10, 10] for x in range(0, 72, 12)]
    lower = [[x, 20, x + 10, 30] for x in range(0, 72, 12)]
    boxes = lower[::-1] + upper                                   # any order in, same answer out
    codes = anterior12_codes(boxes)
    assert codes == list(ANTERIOR_LOWER[::-1]) + list(ANTERIOR_UPPER) and anterior12_codes(upper) is None
    molar, far = [100, 0, 110, 10], [200, 0, 210, 10]
    c, b = other_teeth(boxes, codes, [13, 24, 25, 24], [upper[0], molar, far, far], [0.9, 0.9, 0.3, 0.8], 0.3, 0.5)
    assert c == [24] and b == [molar]                             # low score dropped, repeated 24 dropped
    assert other_teeth(boxes, codes, [12], [upper[0]], [0.9], 0.3, 0.5) is None      # model says 12 where layout says 13
    for f in sorted((MODULE_ROOT / "configs/data").glob("*.yaml")):
        if "extras" in f.read_text():
            u = load_union_cfg(f, {"data_root": "/D", "exp_root": "/E"})
            assert u["base_yolo_root"].startswith("/D/") and all(e["preds_from"].startswith("/E/") for e in u["extras"])


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
