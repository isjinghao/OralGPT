#!/usr/bin/env python
"""Step 8 -- train the condition probes on every cached feature file and write the comparison table.

    python scripts/08_condition_probe.py --probe configs/probe/condition_box.yaml

Writes <out_root>/<probe>/{results.json, report.md}.
"""
import argparse
import json
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from encprobe.config import all_encoder_cfgs, feature_file, load_encoder_cfg, load_probe_cfg
from encprobe.probe import run_condition_probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    args = ap.parse_args()
    probe = load_probe_cfg(args.probe)
    out, dev = Path(probe["out_root"]) / probe["name"], "cuda" if torch.cuda.is_available() else "cpu"
    rows, floor, classes = [], None, None
    for size in probe["image_sizes"]:
        for f in all_encoder_cfgs():
            enc = load_encoder_cfg(f)
            ff = feature_file(probe, enc["name"], size)
            if not ff.exists():
                continue
            d = torch.load(ff, map_location="cpu", weights_only=False)
            classes, mods = d["classes"], sorted(d["classes"])
            fit = lambda mode: {m: run_condition_probe(d["train"], d["val"], d["test"], i, classes[m], mode, probe["probe"], dev)
                                for i, m in enumerate(mods)}
            if floor is None:                         # geometry does not depend on the encoder
                floor = fit("geom")
                print("geom floor", {m: v["test"]["balanced"] for m, v in floor.items()}, flush=True)
            for mode in ("linear", "mlp"):
                per = fit(mode)
                rows.append(dict(encoder=enc["name"], backbone=enc["backbone"], pretrain=enc["pretrain"], size=size, mode=mode,
                                 dim=int(d["train"]["feat"].shape[1]), params_m=d["params_m"], per_modality=per))
                print(enc["name"], size, mode, {m: v["test"]["balanced"] for m, v in per.items()}, flush=True)
    (out / "results.json").write_text(json.dumps(dict(classes=classes, floor=floor, rows=rows), indent=1))

    mods = sorted(classes)
    mean = lambda per, k: round(sum(per[m]["test"][k] for m in mods) / len(mods), 4)
    line = lambda head, per, k: f"| {head} | " + " | ".join(str(per[m]["test"][k]) for m in mods) + f" | {mean(per, k)} |"
    md = [f"# Frozen-encoder probe: {probe['name']}", "",
          "Each ground-truth condition box of the multimodal detection set is RoI-pooled from the frozen encoder's last "
          "feature map and classified among the condition classes of its modality. Bench test split, epoch selected on val.", "",
          "Classes: " + "; ".join(f"**{m}** ({floor[m]['n_train']} train / {floor[m]['n_test']} test boxes): " + ", ".join(classes[m]) for m in mods), ""]
    for metric, title in (("balanced", "balanced accuracy (mean per-class recall)"), ("macro_f1", "macro F1"), ("acc", "accuracy")):
        for size in probe["image_sizes"]:
            for mode in ("linear", "mlp"):
                md += [f"## {title}: {mode}, input {size}", "", "| encoder | pretraining | params (M) | dim | " + " | ".join(mods) + " | mean |",
                       "|---|---|---|---|" + "---|" * (len(mods) + 1)]
                md += [line(f"{r['encoder']} | {r['pretrain']} | {r['params_m']} | {r['dim']}", r["per_modality"], metric)
                       for r in rows if r["size"] == size and r["mode"] == mode]
                md += [line("box position and size only | – | – | 4", floor, metric), ""]
    (out / "report.md").write_text("\n".join(md))
    print("report:", out / "report.md")


if __name__ == "__main__":
    main()
