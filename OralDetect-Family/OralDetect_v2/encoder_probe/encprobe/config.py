"""Config loading. Configs hold relative paths; absolute roots come from OralDetect_v2/root_paths.yaml."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

MODULE_ROOT = Path(__file__).resolve().parents[1]
ROOT_PATHS_FILE = Path(os.environ.get("ORALDETECT_V2_ROOT_PATHS", MODULE_ROOT.parent / "root_paths.yaml"))


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = MODULE_ROOT / path
    return yaml.safe_load(open(path))


def load_roots() -> dict:
    return load_yaml(ROOT_PATHS_FILE)


def _join(cfg: dict, key: str, root: str) -> None:
    if key in cfg:
        if Path(cfg[key]).is_absolute():
            raise ValueError(f"`{key}: {cfg[key]}` is absolute; configs take relative paths")
        cfg[key] = str(Path(root) / cfg[key])


def load_encoder_cfg(path, roots: dict | None = None) -> dict:
    cfg, roots = load_yaml(path), roots or load_roots()
    _join(cfg, "weights", roots["exp_root"])
    return cfg


def load_probe_cfg(path, roots: dict | None = None) -> dict:
    cfg, roots = load_yaml(path), roots or load_roots()
    _join(cfg, "yolo_root", roots["data_root"])
    _join(cfg, "bench_root", roots["data_root"])
    for v in cfg.get("coco", {}).values():
        _join(v, "images", roots["data_root"]), _join(v, "ann", roots["data_root"])
    _join(cfg, "out_root", roots["exp_root"])
    return cfg


def all_encoder_cfgs() -> list[Path]:
    return sorted((MODULE_ROOT / "configs/encoders").glob("*.yaml"))


def feature_file(probe: dict, encoder: str, size: int) -> Path:
    return Path(probe["out_root"]) / probe["name"] / "features" / f"{encoder}_{size}.pt"
