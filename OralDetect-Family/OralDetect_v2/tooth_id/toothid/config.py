"""Config loading and path resolution.

Configs hold RELATIVE paths only. The absolute roots live in one git-ignored file,
`OralDetect_v2/root_paths.yaml`, and are joined on here -- nowhere else.

    data config    raw_root, yolo_root   -> relative to `data_root`
    model config   out_root, weights     -> relative to `exp_root`
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

MODULE_ROOT = Path(__file__).resolve().parents[1]          # .../OralDetect_v2/tooth_id
V2_ROOT = MODULE_ROOT.parent                               # .../OralDetect_v2
ROOT_PATHS_FILE = Path(os.environ.get("ORALDETECT_V2_ROOT_PATHS", V2_ROOT / "root_paths.yaml"))

_DATA_KEYS = ("raw_root", "yolo_root")
_MODEL_KEYS = ("out_root", "weights")


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():       # config paths are relative to the module
        path = MODULE_ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return cfg


def load_roots() -> dict:
    if not ROOT_PATHS_FILE.exists():
        raise FileNotFoundError(f"{ROOT_PATHS_FILE} not found -- copy root_paths.example.yaml to it and fill in")
    return load_yaml(ROOT_PATHS_FILE)


def _resolve(cfg: dict, keys: tuple[str, ...], root_key: str, roots: dict | None) -> dict:
    roots = roots if roots is not None else load_roots()
    for k in keys:
        if Path(cfg[k]).is_absolute():
            raise ValueError(f"`{k}: {cfg[k]}` is absolute; configs take paths relative to `{root_key}`")
        cfg[k] = str(Path(roots[root_key]) / cfg[k])
    return cfg


def load_data_cfg(path: str | Path, roots: dict | None = None) -> dict:
    return _resolve(load_yaml(path), _DATA_KEYS, "data_root", roots)


def load_model_cfg(path: str | Path, roots: dict | None = None) -> dict:
    return _resolve(load_yaml(path), _MODEL_KEYS, "exp_root", roots)


def run_dir(model_cfg: dict) -> Path:
    """<exp_root>/<out_root>/<name> -- the one directory a run writes to."""
    return Path(model_cfg["out_root"]) / model_cfg["name"]


def load_external_cfg(path: str | Path, roots: dict | None = None) -> dict:
    """Config of an external (test-only) set: a list of `sources`, each with `images` and `annotations`."""
    cfg, roots = load_yaml(path), roots if roots is not None else load_roots()
    for src in cfg["sources"]:
        _resolve(src, ("images", "annotations"), "data_root", roots)
    return cfg


def load_unlabeled_cfg(path: str | Path, roots: dict | None = None) -> dict:
    """Config of image sets to be numbered: `datasets`, each with a `prefix` and a list of `dirs`."""
    cfg, roots = load_yaml(path), roots if roots is not None else load_roots()
    for ds in cfg["datasets"]:
        for d in ds["dirs"]:
            if Path(d).is_absolute():
                raise ValueError(f"`{d}` is absolute; configs take paths relative to `data_root`")
        ds["dirs"] = [str(Path(roots["data_root"]) / d) for d in ds["dirs"]]
    return cfg


def load_union_cfg(path: str | Path, roots: dict | None = None) -> dict:
    """Data config of a base FDI dataset plus `extras` (sets with human boxes but incomplete numbers,
    completed with a run's predictions): adds `base_yolo_root` (data_root) and each `preds_from` (exp_root)."""
    roots = roots if roots is not None else load_roots()
    cfg = _resolve(load_data_cfg(path, roots), ("base_yolo_root",), "data_root", roots)
    for ex in cfg["extras"]:
        _resolve(ex, ("preds_from",), "exp_root", roots)
    return cfg
