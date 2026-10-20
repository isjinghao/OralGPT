#!/usr/bin/env python3
"""Plot raw SFT training and validation losses from LLaMA-Factory state."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/OralGPT/OralGPT-CMF/sft/sft_output/qwen3_5_9b_lora_turn_level")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_path = args.output or output_dir / "training_validation_curves.pdf"
    state = json.loads((output_dir / "trainer_state.json").read_text(encoding="utf-8"))
    history = state["log_history"]
    train = [item for item in history if "loss" in item]
    validation = [item for item in history if "eval_loss" in item]
    if not train or not validation:
        raise ValueError("trainer_state.json does not contain both training and validation losses")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), constrained_layout=True)
    line_style = {"color": "#1f77b4", "linewidth": 1.8, "markersize": 3.0}
    panels = (
        (axes[0], train, "loss", "Training loss"),
        (axes[1], validation, "eval_loss", "Validation loss"),
    )
    for axis, records, metric, label in panels:
        axis.plot(
            [record["step"] for record in records],
            [record[metric] for record in records],
            marker="o",
            label=label,
            **line_style,
        )
        axis.set_xlabel("Training Step")
        axis.set_ylabel("Cross-entropy loss")
        axis.set_xlim(0, state["global_step"] + 50)
        axis.xaxis.set_major_locator(MultipleLocator(500))
        axis.grid(True, color="#b0b0b0", alpha=0.55, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()