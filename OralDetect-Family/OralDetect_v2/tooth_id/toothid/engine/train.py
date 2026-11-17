"""Training. A thin wrapper: every hyperparameter lives in the model yaml, none here."""
from __future__ import annotations

from pathlib import Path

from ..config import run_dir


def assert_label_safe(train_args: dict) -> None:
    """A horizontal flip swaps left and right quadrants (16 becomes 26) while the label stays; a
    vertical flip swaps the arches. Either one trains the model on wrong tooth IDs."""
    for k in ("fliplr", "flipud"):
        if float(train_args.get(k, 0.5 if k == "fliplr" else 0.0)) != 0.0:
            raise ValueError(f"train.{k} must be 0.0 for FDI numbering (set it explicitly in the model yaml)")


def train(model_cfg: dict, dataset_yaml: str | Path, overrides: dict | None = None):
    from ultralytics import YOLO

    args = {**model_cfg["train"], **(overrides or {})}
    assert_label_safe(args)
    rd = run_dir(model_cfg)
    model = YOLO(model_cfg["weights"])
    return model.train(data=str(dataset_yaml), project=str(rd.parent), name=rd.name, exist_ok=True, **args)
