"""Data-free tests:  python -m pytest tests -q"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from encprobe.config import MODULE_ROOT, all_encoder_cfgs, load_encoder_cfg, load_probe_cfg  # noqa: E402
from encprobe.encoders import _fb_to_timm  # noqa: E402
from encprobe.probe import run_probe  # noqa: E402


def test_configs_are_relative():
    roots = {"data_root": "/D", "exp_root": "/E"}
    for f in sorted((MODULE_ROOT / "configs").rglob("*.yaml")):
        assert "/ix/" not in f.read_text() and "/ihome/" not in f.read_text(), f
    for f in all_encoder_cfgs():
        c = load_encoder_cfg(f, roots)
        assert c["name"] == f.stem and c.get("weights", "/E/").startswith("/E/")
    p = load_probe_cfg("configs/probe/tooth_fdi.yaml", roots)
    assert p["yolo_root"].startswith("/D/") and p["out_root"].startswith("/E/")


def test_convnext_key_map():
    assert _fb_to_timm("downsample_layers.0.0.weight") == "stem.0.weight"
    assert _fb_to_timm("downsample_layers.2.1.bias") == "stages.2.downsample.1.bias"
    assert _fb_to_timm("stages.1.3.pwconv2.weight") == "stages.1.blocks.3.mlp.fc2.weight"
    assert _fb_to_timm("norm.weight") is None


def test_probe_learns_separable_features():
    g = torch.Generator().manual_seed(0)
    def split(n):
        y = torch.randint(0, 32, (n,), generator=g)
        return dict(feat=(torch.eye(32)[y] + 0.05 * torch.randn(n, 32, generator=g)).half(), geom=torch.rand(n, 4, generator=g),
                    cls=y, view=torch.randint(0, 5, (n,), generator=g))
    hp = dict(epochs=20, batch=256, lr=0.01, weight_decay=0.0, mlp_hidden=32, seed=0)
    r = run_probe(split(2000), split(300), split(300), "linear", hp, "cpu")
    assert r["test"]["fdi"] > 0.95 and r["test"]["position"] >= r["test"]["fdi"]


def test_condition_classes_and_probe():
    from encprobe.data import condition_classes
    from encprobe.probe import run_condition_probe

    def coco(pairs):
        return dict(images=[dict(id=i, modality=m) for i, (m, _) in enumerate(pairs)],
                    categories=[dict(id=j, name=n) for j, n in enumerate(sorted({n for _, n in pairs}))],
                    annotations=[dict(image_id=i, category_id=sorted({n for _, n in pairs}).index(n)) for i, (_, n) in enumerate(pairs)])
    tr = coco([("pano", "caries"), ("pano", "filling"), ("pano", "tooth 11 (x)"), ("pano", "plaque"), ("histo", "tumor"), ("intra", "caries")])
    te = coco([("pano", "caries"), ("pano", "filling"), ("pano", "tooth 11 (x)"), ("histo", "tumor"), ("intra", "caries")])
    assert condition_classes(tr, te, [r"^tooth \d\d "]) == {"pano": ["caries", "filling"]}

    g = torch.Generator().manual_seed(0)
    def split(n):
        y, mod = torch.randint(0, 3, (n,), generator=g), torch.randint(0, 2, (n,), generator=g)
        return dict(feat=(torch.eye(3)[y] + 0.05 * torch.randn(n, 3, generator=g)).half(), geom=torch.rand(n, 4, generator=g), cls=y, mod=mod)
    hp = dict(epochs=20, batch=256, lr=0.01, weight_decay=0.0, mlp_hidden=32, seed=0)
    r = run_condition_probe(split(2000), split(300), split(300), 1, ["a", "b", "c"], "linear", hp, "cpu")
    assert r["test"]["balanced"] > 0.95 and set(r["test"]["recall"]) == {"a", "b", "c"}
    assert run_condition_probe(split(2000), split(300), split(300), 1, ["a", "b", "c"], "geom", hp, "cpu")["test"]["balanced"] < 0.6
