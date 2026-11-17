#!/usr/bin/env python
# Evaluate an OralDetect checkpoint on one or more benches.
#
#     torchrun --nproc_per_node=4 run_eval.py --config launch_bash/eval_oraldetect.yaml
#     run_eval.py --config ... --dry-run          validate + key-check, score nothing
#     run_eval.py --config ... --bench main       just one bench from the list
#
# Mirrors run_finetune.py: the .sh allocates GPUs, this does path validation, the resolved-config
# summary, the load_from key check, the scoring loop, and the cross-bench summary table.
#
# There is no eval config. `oraldetect_finetune.py` already carries a complete
# test_pipeline / test_dataloader / test_evaluator, so reusing it is not a shortcut -- it is the
# only way to guarantee the model being scored is the same one that was trained. All this file
# redirects is the annotation file, the image root, and where predictions land.
#
# Several benches in one job is safe: mmengine 0.10.7 guards `init_dist` with
# `if self.distributed and not is_distributed()`, so building one Runner per bench in the same
# torchrun process does not re-initialise the process group.
from __future__ import annotations

import argparse
import copy
import json
import os
import os.path as osp
import sys

import yaml

MODALITIES = ["intraoral", "panoramic", "periapical", "cytology", "histology"]


def is_main() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def say(*a) -> None:
    if is_main():
        print(*a, flush=True)


def resolve(y: dict, yaml_path: str) -> dict:
    """Make every path in the yaml absolute, anchored at the repo root.

    The launcher chdir's into `paths.wedetect` before building anything, so a relative path left
    as-is would resolve against the framework directory rather than the repo. Anchoring here, once,
    is what lets the shipped yaml use short relative paths and still work from any cwd.

    The repo root is two levels above the yaml (repo/launch_bash/x.yaml).
    """
    root = osp.dirname(osp.dirname(osp.abspath(yaml_path)))
    KEYS = {"wedetect", "config", "init", "checkpoint", "text_tower", "work_dir", "out_dir",
            "data_root", "train_ann", "val_ann", "class_names", "class_texts", "ann"}

    def fix(v):
        return v if osp.isabs(v) else osp.normpath(osp.join(root, v))

    for sec in ("paths", "data"):
        for k, v in (y.get(sec) or {}).items():
            if k in KEYS and isinstance(v, str):
                y[sec][k] = fix(v)
    for b in (y.get("benches") or []):
        for k in ("ann", "data_root"):
            if k in b and isinstance(b[k], str):
                b[k] = fix(b[k])
    return y


def load_yaml(path: str) -> dict:
    if not osp.isfile(path):
        sys.exit(f"FATAL: no such yaml -- {path}")
    y = resolve(yaml.safe_load(open(path)), path)

    problems = []
    for section, key, kind in [("paths", "wedetect", "dir"), ("paths", "config", "file"),
                               ("paths", "checkpoint", "file"), ("data", "data_root", "dir"),
                               ("data", "class_names", "file"), ("data", "class_texts", "file")]:
        try:
            p = y[section][key]
        except (KeyError, TypeError):
            problems.append(f"{section}.{key} is not set in the yaml")
            continue
        if not (osp.isdir(p) if kind == "dir" else osp.isfile(p)):
            problems.append(f"{section}.{key}: no such {kind} -- {p}")

    benches = y.get("benches") or []
    if not benches:
        problems.append("benches: the list is empty -- nothing to evaluate")
    seen = set()
    for i, b in enumerate(benches):
        tag = b.get("name", f"benches[{i}]")
        if tag in seen:
            problems.append(f"benches: duplicate name {tag!r} -- results would overwrite")
        seen.add(tag)
        if not b.get("ann"):
            problems.append(f"{tag}: no `ann` set")
        elif not osp.isfile(b["ann"]):
            problems.append(f"{tag}: no such ann file -- {b['ann']}")
        root = b.get("data_root", y.get("data", {}).get("data_root"))
        if root and not osp.isdir(root):
            problems.append(f"{tag}: no such data_root -- {root}")
    if problems:
        sys.exit("FATAL: " + "\nFATAL: ".join(problems))
    return y


def base_cfg(y: dict):
    """The finetune config, with the model/vocab bits from the yaml applied. Bench-independent."""
    from mmengine.config import Config

    cfg = Config.fromfile(y["paths"]["config"])
    d = y["data"]

    classes = json.load(open(d["class_names"]))
    if not isinstance(classes, list) or not classes:
        sys.exit(f"FATAL: {d['class_names']} is not a non-empty list of class names")
    n = len(classes)
    if n > 256:
        sys.exit(f"FATAL: {n} classes > 256 -- the classifier branch changes shape past that and "
                 "the checkpoint's would be dropped. See the config header.")

    cfg.merge_from_dict({
        "load_from": y["paths"]["checkpoint"],
        "model.num_train_classes": n,
        # optional: the config bakes in an absolute path to the DentalBERT text tower, which is
        # only right on the machine it was written on. Anyone else must be able to redirect it.
        **({"model.backbone.text_model.model_name": y["paths"]["text_tower"]}
           if y["paths"].get("text_tower") else {}),
        "model.num_test_classes": n,
        "model.bbox_head.head_module.num_classes": n,
        "model.train_cfg.assigner.num_classes": n,
        "val_dataloader.dataset.class_text_path": d["class_texts"],
        "val_dataloader.dataset.dataset.metainfo": dict(classes=tuple(classes)),
        "val_dataloader.batch_size": int(y.get("eval", {}).get("batch_per_gpu", 1)),
    })
    cfg.resume = False
    return cfg, n


def bench_cfg(cfg, y: dict, bench: dict):
    """Point a copy of the base config at one bench."""
    c = copy.deepcopy(cfg)
    ann = bench["ann"]
    root = bench.get("data_root", y["data"]["data_root"])
    out = osp.join(y["paths"]["out_dir"], bench["name"])

    c.val_dataloader.dataset.dataset.ann_file = ann
    c.val_dataloader.dataset.dataset.data_root = root
    if y["data"].get("evaluator", "per_modality") == "coco":
        c.val_evaluator = dict(type="mmdet.CocoMetric", ann_file=ann, metric="bbox")
    else:
        c.val_evaluator = dict(type="PerModalityCocoMetric", ann_file=ann, metric="bbox")
    if y.get("eval", {}).get("dump_preds", True):
        c.val_evaluator["outfile_prefix"] = osp.join(out, "preds")
    c.test_dataloader = c.val_dataloader
    c.test_evaluator = c.val_evaluator
    c.work_dir = out
    return c


def check_load(cfg, strict: bool) -> None:
    """Diff the built model against the checkpoint before scoring anything."""
    import torch
    from mmdet.registry import MODELS
    from mmengine.registry import DefaultScope

    with DefaultScope.overwrite_default_scope(cfg.get("default_scope", "mmdet")):
        model = MODELS.build(cfg.model)
    msd = model.state_dict()
    ck = torch.load(cfg.load_from, map_location="cpu", weights_only=False)
    csd = ck.get("state_dict", ck)

    mk, ckk = set(msd), set(csd)
    missing, unexpected = sorted(mk - ckk), sorted(ckk - mk)
    shape = sorted(k for k in (mk & ckk) if tuple(msd[k].shape) != tuple(csd[k].shape))
    say(f"  checkpoint  : {len(mk)} model params vs {len(ckk)} in checkpoint")
    say(f"                missing {len(missing)} | unexpected {len(unexpected)} | "
        f"shape-mismatch {len(shape)}")
    for tag, ks in (("missing", missing), ("unexpected", unexpected), ("shape", shape)):
        for k in ks[:8]:
            say(f"                  [{tag}] {k}")
        if len(ks) > 8:
            say(f"                  [{tag}] ... and {len(ks) - 8} more")
    del model, msd, ck, csd

    if missing or unexpected or shape:
        if strict:
            sys.exit("FATAL: this checkpoint does not match the model built from `config`. "
                     "Scoring it would report numbers for a partly-random model. Set "
                     "strict_load: false only if you mean to.")
        say("  ⚠️  strict_load is false -- scoring an INCOMPLETELY loaded model.")
    else:
        say("  checkpoint  : CLEAN ✅")


def report(name: str, m: dict) -> None:
    per = [(k.split("/")[-1][:-4], v) for k, v in m.items()
           if k.endswith("_mAP") and k.split("/")[-1][:-4] in MODALITIES]
    say(f"\n--- {name} ---")
    for mod, v in sorted(per, key=lambda kv: MODALITIES.index(kv[0])):
        say(f"    {mod:<12} {v:.4f}")
    if "coco/macro_mAP" in m:
        say(f"    {'MACRO-5':<12} {m['coco/macro_mAP']:.4f}   <- report")
    if "coco/macro4_mAP" in m:
        say(f"    {'MACRO-4':<12} {m['coco/macro4_mAP']:.4f}   <- select")
    if "coco/bbox_mAP" in m:
        say(f"    {'bbox_mAP':<12} {m['coco/bbox_mAP']:.4f}   (class-macro; archive only, do not report)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate an OralDetect checkpoint on one or more benches.")
    ap.add_argument("--config", required=True, help="path to the eval yaml")
    ap.add_argument("--bench", default=None, help="run only this bench from the yaml's list")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate, print the resolved config and the key diff, then stop")
    args = ap.parse_args()

    y = load_yaml(args.config)
    sys.path.insert(0, y["paths"]["wedetect"])
    os.chdir(y["paths"]["wedetect"])
    import wedetect  # noqa: F401

    benches = y["benches"]
    if args.bench:
        benches = [b for b in benches if b.get("name") == args.bench]
        if not benches:
            sys.exit(f"FATAL: no bench named {args.bench!r} in {args.config}")

    cfg, n_cls = base_cfg(y)

    world = int(os.environ.get("WORLD_SIZE", 1))
    want = int(y.get("eval", {}).get("gpus", world))
    if world != want and not args.dry_run:
        sys.exit(f"FATAL: yaml says eval.gpus={want} but torchrun launched {world} processes.")

    say("=" * 78)
    say("OralDetect eval")
    say("=" * 78)
    say(f"  checkpoint  : {cfg.load_from}")
    say(f"  config      : {y['paths']['config']}")
    say(f"  vocab       : {n_cls} classes  ({y['data']['class_names']})")
    say(f"  prompts     : {y['data']['class_texts']}")
    say(f"  evaluator   : {y['data'].get('evaluator', 'per_modality')}")
    say(f"  resolution  : {cfg.img_scale}   batch {cfg.val_dataloader.batch_size}/gpu x {world} gpu")
    say(f"  out dir     : {y['paths']['out_dir']}")
    say(f"  benches     : {', '.join(b['name'] for b in benches)}")
    check_load(cfg, strict=bool(y.get("strict_load", True)))

    if args.dry_run:
        for b in benches:
            c = bench_cfg(cfg, y, b)
            say(f"    {b['name']:<22} ann={c.val_dataloader.dataset.dataset.ann_file}")
            say(f"    {'':<22} root={c.val_dataloader.dataset.dataset.data_root}")
            say(f"    {'':<22} out={c.work_dir}")
        say("\n[--dry-run] validated, nothing scored.")
        return 0

    from mmengine.runner import Runner

    results = {}
    for b in benches:
        say(f"\n{'=' * 78}\nscoring: {b['name']}\n{'=' * 78}")
        c = bench_cfg(cfg, y, b)
        os.makedirs(c.work_dir, exist_ok=True)
        try:
            results[b["name"]] = Runner.from_cfg(c).test()
        except Exception as e:                      # one bad bench must not lose the others
            say(f"  !! {b['name']} FAILED: {type(e).__name__}: {e}")
            results[b["name"]] = None

    say("\n" + "=" * 78)
    say("SUMMARY")
    say("=" * 78)
    for name, m in results.items():
        if m:
            report(name, m)
        else:
            say(f"\n--- {name} ---\n    FAILED (see above)")
    say(f"\n  predictions under {y['paths']['out_dir']}/<bench>/preds.bbox.json")
    say("  offline re-score: tools/per_modality_eval.py <gt.json> <preds.bbox.json>")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
