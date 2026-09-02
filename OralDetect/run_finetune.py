#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

import yaml

REQUIRED_PATHS = [
    ("paths", "wedetect", "dir"),
    ("paths", "config", "file"),
    ("paths", "init", "file"),
    ("data", "data_root", "dir"),
    ("data", "train_ann", "file"),
    ("data", "val_ann", "file"),
    ("data", "class_names", "file"),
    ("data", "class_texts", "file"),
]


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
    with open(path) as f:
        cfg = resolve(yaml.safe_load(f), path)
    missing = []
    for section, key, kind in REQUIRED_PATHS:
        try:
            p = cfg[section][key]
        except (KeyError, TypeError):
            missing.append(f"{section}.{key} is not set in the yaml")
            continue
        ok = osp.isdir(p) if kind == "dir" else osp.isfile(p)
        if not ok:
            missing.append(f"{section}.{key}: no such {kind} -- {p}")
    if missing:
        sys.exit("FATAL: " + "\nFATAL: ".join(missing))
    return cfg


def retarget_text_sampling(node, n: int) -> int:
    """Point every RandomLoadText in `node` at an n-class vocabulary. Returns how many it found.

    RandomLoadText pads each sample's prompt list to `max_num_samples`, and the config bakes in 87
    -- the vocabulary this checkpoint was trained on. Left alone it hands the model 87 prompts
    while `num_train_classes` says n, and the first batch dies on the shape mismatch. At n=1 the
    padding *is* the batch: one real prompt and 86 empty strings.
    """
    hit = 0
    if isinstance(node, dict):
        if node.get("type") == "RandomLoadText":
            node["max_num_samples"] = n
            node["num_neg_samples"] = (n, n)      # sample the whole vocabulary, as 87 did for 87
            hit += 1
        for v in node.values():
            hit += retarget_text_sampling(v, n)
    elif isinstance(node, (list, tuple)):
        for v in node:
            hit += retarget_text_sampling(v, n)
    return hit


def build_cfg(y: dict):
    """The mmengine Config, with every path and hyper-parameter from the yaml merged over it."""
    from mmengine.config import Config

    cfg = Config.fromfile(y["paths"]["config"])
    d, t = y["data"], y.get("train", {})

    classes = json.load(open(d["class_names"]))
    if not isinstance(classes, list) or not classes:
        sys.exit(f"FATAL: {d['class_names']} is not a non-empty list of class names")
    n = len(classes)
    # YOLOWorldHeadModule sizes its classifier as max(in_channels[0], num_classes) = max(256, n),
    # and the class prototypes come from the text tower rather than a learned per-class matrix.
    # So any vocabulary of <=256 classes keeps the checkpoint's exact weight shapes. Past that the
    # cls_preds convs change and load_from drops them -- a different experiment, not a finetune.
    if n > 256:
        sys.exit(f"FATAL: {n} classes > 256 -- the classifier branch would change shape and "
                 f"load_from would drop it. See {y['paths']['config']}.")
    metainfo = dict(classes=tuple(classes))

    over = {
        "load_from": y["paths"]["init"],
        "model.num_train_classes": n,
        # optional: the config bakes in an absolute path to the DentalBERT text tower, which is
        # only right on the machine it was written on. Anyone else must be able to redirect it.
        **({"model.backbone.text_model.model_name": y["paths"]["text_tower"]}
           if y["paths"].get("text_tower") else {}),
        "model.num_test_classes": n,
        "model.bbox_head.head_module.num_classes": n,
        "model.train_cfg.assigner.num_classes": n,
        "train_dataloader.dataset.class_text_path": d["class_texts"],
        "train_dataloader.dataset.dataset.metainfo": metainfo,
        "train_dataloader.dataset.dataset.data_root": d["data_root"],
        "train_dataloader.dataset.dataset.ann_file": d["train_ann"],
        "val_dataloader.dataset.class_text_path": d["class_texts"],
        "val_dataloader.dataset.dataset.metainfo": metainfo,
        "val_dataloader.dataset.dataset.data_root": d["data_root"],
        "val_dataloader.dataset.dataset.ann_file": d["val_ann"],
        "val_evaluator.ann_file": d["val_ann"],
    }
    if "lr" in t:
        over["optim_wrapper.optimizer.lr"] = float(t["lr"])
    if "epochs" in t:
        over["train_cfg.max_epochs"] = int(t["epochs"])
        over["param_scheduler.1.end"] = int(t["epochs"])
    if "batch_per_gpu" in t:
        over["train_dataloader.batch_size"] = int(t["batch_per_gpu"])
        over["optim_wrapper.optimizer.batch_size_per_gpu"] = int(t["batch_per_gpu"])

    if d.get("evaluator", "per_modality") == "coco":
        # _delete_ so the PerModalityCocoMetric keys do not leak into mmdet.CocoMetric
        cfg.val_evaluator = dict(type="mmdet.CocoMetric", ann_file=d["val_ann"], metric="bbox")
        cfg.default_hooks.checkpoint.save_best = "coco/bbox_mAP"
        over.pop("val_evaluator.ann_file")
    cfg.merge_from_dict(over)
    # The pipelines are lists of dicts, so merge_from_dict cannot reach into them by name; and
    # train_pipeline / train_pipeline_stage2 are separate objects from the dataloader's copy.
    n_text = sum(retarget_text_sampling(cfg.get(k), n) for k in
                 ("train_dataloader", "train_dataset", "train_pipeline", "train_pipeline_stage2",
                  "text_transform", "custom_hooks"))
    if not n_text:
        sys.exit(f"FATAL: no RandomLoadText found in {y['paths']['config']} -- the prompt sampler "
                 f"moved, and the vocabulary size would stay at the config's baked-in value.")
    cfg.test_dataloader = cfg.val_dataloader
    cfg.test_evaluator = cfg.val_evaluator
    cfg.work_dir = y["paths"]["work_dir"]
    return cfg, n, n_text


def check_load(cfg, strict: bool) -> None:
    """Diff the built model against the checkpoint. This is the whole point of the wrapper."""
    import torch
    from mmdet.registry import MODELS          # WeDetect registers into mmdet's registry
    from mmengine.registry import DefaultScope

    with DefaultScope.overwrite_default_scope(cfg.get("default_scope", "mmdet")):
        model = MODELS.build(cfg.model)
    msd = model.state_dict()
    ck = torch.load(cfg.load_from, map_location="cpu", weights_only=False)
    csd = ck.get("state_dict", ck)

    mk, ck_keys = set(msd), set(csd)
    missing = sorted(mk - ck_keys)          # in the model, absent from the ckpt -> RANDOM
    unexpected = sorted(ck_keys - mk)       # in the ckpt, absent from the model -> DROPPED
    shape = sorted(k for k in (mk & ck_keys) if tuple(msd[k].shape) != tuple(csd[k].shape))

    say(f"  load_from   : {len(mk)} model params vs {len(ck_keys)} in checkpoint")
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
            sys.exit("FATAL: load_from would not land cleanly. Either the finetune config's "
                     "architecture drifted from the checkpoint, or you meant to change it -- in "
                     "which case set strict_load: false in the yaml and say so in the write-up.")
        say("  ⚠️  strict_load is false -- continuing with an INCOMPLETE load.")
    else:
        say("  load_from   : CLEAN ✅")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Finetune OralDetect from our own trained detector.")
    ap.add_argument("--config", required=True, help="path to the finetune yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate, print the resolved config and the key diff, then stop")
    args = ap.parse_args()

    y = load_yaml(args.config)
    sys.path.insert(0, y["paths"]["wedetect"])
    os.chdir(y["paths"]["wedetect"])          # the configs use paths relative to the framework root
    import wedetect  # noqa: F401             # registers every custom module

    cfg, n_cls, n_text = build_cfg(y)

    # torchrun's world size is the truth; the yaml value only exists so the .sh can read it.
    # Not enforced under --dry-run: the whole point of a dry run is to validate the config on one
    # cheap CPU process, without allocating the GPUs the real run wants.
    world = int(os.environ.get("WORLD_SIZE", 1))
    want = int(y.get("train", {}).get("gpus", world))
    if world != want and not args.dry_run:
        sys.exit(f"FATAL: yaml says train.gpus={want} but torchrun launched {world} processes. "
                 f"Fix --nproc_per_node in the .sh or train.gpus in the yaml.")

    os.makedirs(cfg.work_dir, exist_ok=True)
    # mmengine's load_or_resume(): resume=True *with load_from set* resumes from load_from -- the
    # released weights -- and every epoch already trained is silently thrown away. Resume has to
    # mean resume, so load_from is repointed at whatever last_checkpoint names.
    last = osp.join(cfg.work_dir, "last_checkpoint")
    resume_ckpt = open(last).read().strip() if osp.isfile(last) else None
    if resume_ckpt and not osp.isfile(resume_ckpt):
        moved = osp.join(cfg.work_dir, osp.basename(resume_ckpt))   # work_dir copied or renamed
        if not osp.isfile(moved):
            sys.exit(f"FATAL: {last} names {resume_ckpt}, which does not exist. Point it at a "
                     f"checkpoint that does, or delete it to train from {cfg.load_from} again.")
        resume_ckpt = moved
    resume = resume_ckpt is not None
    cfg.resume = resume
    if resume:
        cfg.load_from = resume_ckpt

    say("=" * 78)
    say("OralDetect finetune")
    say("=" * 78)
    say(f"  init        : {y['paths']['init']}")
    say(f"  config      : {y['paths']['config']}")
    say(f"  train ann   : {cfg.train_dataloader.dataset.dataset.ann_file}")
    say(f"  val ann     : {cfg.val_dataloader.dataset.dataset.ann_file}")
    say(f"  image root  : {cfg.train_dataloader.dataset.dataset.data_root}")
    say(f"  vocab       : {n_cls} classes  ({y['data']['class_names']})")
    say(f"  prompts     : {n_cls} per sample  ({n_text} RandomLoadText retargeted from the "
        f"config's 87)")
    say(f"  evaluator   : {cfg.val_evaluator['type']}  -> save_best "
        f"{cfg.default_hooks.checkpoint.save_best}")
    say(f"  resolution  : {cfg.img_scale}   mosaic: "
        f"{'ON' if any('Mosaic' in str(t.get('type', '')) for t in cfg.train_dataloader.dataset.pipeline) else 'OFF'}")
    say(f"  lr / epochs : {cfg.optim_wrapper.optimizer.lr} / {cfg.train_cfg.max_epochs}")
    say(f"  batch       : {cfg.train_dataloader.batch_size}/gpu x {world} gpu = "
        f"{cfg.train_dataloader.batch_size * world} effective")
    say(f"  work dir    : {cfg.work_dir}")
    say(f"  resume      : {'YES -- ' + resume_ckpt if resume else 'no (fresh run)'}")

    # On resume the weights come from the resume checkpoint, not load_from, so the check is moot.
    if not resume:
        check_load(cfg, strict=bool(y.get("strict_load", True)))
    else:
        say("  load_from   : skipped (resuming -- weights come from the resume checkpoint)")

    if args.dry_run:
        say("\n[--dry-run] validated, nothing trained.")
        return 0

    from mmengine.runner import Runner
    runner = Runner.from_cfg(cfg)
    runner.train()

    say("\n" + "=" * 78)
    for f in sorted(os.listdir(cfg.work_dir)):
        if f.startswith("best_") and f.endswith(".pth"):
            say(f"  best checkpoint: {osp.join(cfg.work_dir, f)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
