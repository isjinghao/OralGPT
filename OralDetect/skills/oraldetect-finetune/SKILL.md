---
name: oraldetect-finetune
description: This skill should be used when the user wants to finetune the OralDetect model on their own dataset — e.g. "finetune OralDetect on my data", "train the dental detector on my data", "run OralDetect on my images". The user provides their dataset folder; validates the COCO format against the loader's real requirements, places the weights, and generates own_*.yaml launchers that are dry-run verified before any GPU is used.
version: 1.0.0
---

# Finetune OralDetect on a user-supplied dataset

Six steps, in order. Each ends in a **Checkpoint**. Do not start step N+1 until step N's
checkpoint holds. A checkpoint that cannot be met is a stop: report what is missing and what the
user has to decide.

## Directories

Ask the user for these three:

```bash
REPO=<the cloned repo — the directory that holds run_finetune.py>
IMAGE_ROOT=<the dataset folder the user gave you>
WEIGHTS=<where the weights are, or will go>
```

Derive the rest. Everything after this point is addressed through these names:

```bash
SKILL=$REPO/skills/oraldetect-finetune            # the scripts this skill runs
DATAS=$REPO/datas                      # generated jsons + vocabulary
LAUNCH=$REPO/launch_bash               # generated yamls
OUT=$REPO/own_finetune                            # checkpoints, logs, predictions
REPORT=$OUT/report                                # every figure and table

mkdir -p "$DATAS" "$LAUNCH" "$REPORT"
```

`IMAGE_ROOT` is both the folder the user handed over and the root that `file_name` in the COCO json
resolves against. It is the one path outside `$REPO`, because the user's images stay where they are.

## What lands where

`$REPORT` is the deliverable — everything a human reads:

```
$REPORT/
├── step1_format.txt            what format the labels were in, what was converted
├── step3_check.txt             the validation report
├── step3_preview.png           annotated samples
├── step3_preview_classes.png   boxes per class
├── step3_preview.ipynb         the notebook behind both figures
├── step6_metrics.txt           precision / recall / F1, before vs after
├── step6_compare.png           ground truth / before / after
└── step6_compare.ipynb         the notebook behind it
```

Every script takes `--save` and writes on every exit path, including failures — a report of *why* a
step blocked is worth keeping. The `.png` names are derived from the `.ipynb` name, so keeping the
names above makes them line up.

Two kinds of file live elsewhere, because they are inputs to training:

| | where | why |
|---|---|---|
| COCO jsons + vocabulary | `$DATAS` | the yaml points at them every run |
| `preds.bbox.json` | `$OUT/eval_before/`, `$OUT/eval_after/` | large, and re-scorable offline |

## Progress

When a checkpoint holds, print its line verbatim before moving on:

```
[STEP 1/6] data check finished
[STEP 2/6] weights check finished
[STEP 3/6] dataset + vocabulary check finished
[STEP 4/6] launcher check finished
[STEP 5/6] baseline scored; training launched
[STEP 6/6] before/after compared — all checks finished
```

If a checkpoint fails, print `[STEP N/6] BLOCKED — <what is missing>` and stop there.

Keep this list current (use the todo tool if available):

```
[ ] 1. labels in COCO; train/val json written to $DATAS; IMAGE_ROOT confirmed
[ ] 2. detector + text tower under $WEIGHTS
[ ] 3. dataset validated; vocabulary approved; boxes visually confirmed
[ ] 4. $LAUNCH/own_finetune.yaml + own_eval.yaml written, every path resolves
[ ] 5. baseline scored; dry-run CLEAN; training past its first validation
[ ] 6. finetuned model scored; before/after table shown
```

## Preflight — the environment

One import line, before anything else. `import wedetect` pulls in `webdataset` at module scope, so
an env that looks complete still dies at the first launcher; the notebook stack is what renders the
two figures.

```bash
python -c "import torch, mmcv, mmengine, mmdet, transformers, webdataset, yaml; \
           import matplotlib, nbformat, nbclient, ipykernel; print('env ok')"
```

Anything missing: `pip install -r $REPO/requirements.txt` — torch and mmcv first, from the repo
README, because they need build flags a requirements file cannot carry. Use one interpreter for
every command in this skill; `run_nb.py` runs the notebooks in whichever one invokes it.

---

## Step 1 — Labels into COCO

Detect the format first; do not assume COCO.

```bash
python $SKILL/scripts/to_coco.py --root "$IMAGE_ROOT" --save "$REPORT/step1_format.txt"
```

Reports `coco` | `yolo` | `voc` | `labelme` | `unknown`.

If the format is `yolo`, `voc` or `labelme`, convert:

```bash
python $SKILL/scripts/to_coco.py --root "$IMAGE_ROOT" --convert --out "$DATAS" --val-frac 0.1
```

Writes `$DATAS/instances_train.json` and `$DATAS/instances_val.json`, and prints the image root it
paired the labels against — set `IMAGE_ROOT` to that. Step 3 writes the vocabulary; this step does
not.

If the format is `coco` and the labels are a single file, split it the same way:

```bash
python $SKILL/scripts/to_coco.py --root "$IMAGE_ROOT" --split <their.json> \
    --out "$DATAS" --val-frac 0.1
```

If the format is `coco` and train/val are already separate, confirm `IMAGE_ROOT` against the json:

```bash
python -c "
import json,sys; print([i['file_name'] for i in json.load(open(sys.argv[1]))['images'][:5]])" <json>
```

`file_name` is relative to `IMAGE_ROOT`. Extend `IMAGE_ROOT` with directory components until
`$IMAGE_ROOT/<file_name>` resolves.

Stop on:
- `unknown` — ask which tool produced the labels, show the user one label file, do not guess.
- label files found, none matching an image — pairing is by filename stem.
- no bounding boxes (image-level labels only) — cannot be used.

> **Checkpoint 1** — `$DATAS/instances_train.json` and `$DATAS/instances_val.json` exist, and
> `IMAGE_ROOT` is set. If the source had polygons, tell the user they became bounding boxes.
>
> Print: `[STEP 1/6] data check finished`

## Step 2 — Weights

```bash
find "$WEIGHTS" -maxdepth 5 \( -name "oraldetect*.pth" -o -name "oralbert" \) 2>/dev/null
```

If they are absent, the user requests access to the gated repo, runs `hf auth login`, then:

```bash
hf download OralGPT/OralDetect-Family --local-dir "$WEIGHTS"
```

| file | use |
|---|---|
| `$WEIGHTS/OralDetect/oraldetect.pth` | converged detector — the default start |
| `$WEIGHTS/OralDetect/oraldetect_init.pth` | towers only, neck+head random — training from scratch |
| `$WEIGHTS/OralCLIP/oralbert` | text tower, required at run time |

> **Checkpoint 2** — both paths exist; `$WEIGHTS/OralCLIP/oralbert` holds `config.json`,
> `model.safetensors` and a tokenizer.
>
> Print: `[STEP 2/6] weights check finished`

## Step 3 — Validate, vocabulary, boxes

```bash
python $SKILL/scripts/check_dataset.py \
    --train "$DATAS/instances_train.json" \
    --val   "$DATAS/instances_val.json" \
    --images "$IMAGE_ROOT" --out "$DATAS" --write-vocab \
    --save "$REPORT/step3_check.txt"
```

Writes `$DATAS/class_names.json` and `$DATAS/class_texts.json`.

Exits non-zero on: category names in val but absent from train; >256 classes; `file_name` that does
not resolve; orphan annotations; bad `category_id`; degenerate boxes. All of these are silent in the
loader — a mismatched category name drops its annotations without an error.

Show the user `$DATAS/class_texts.json` and revise it with them. The text tower embeds these strings
and the embedding is the class prototype.

- expand codes and abbreviations: `DC` → `dental calculus`
- clinical wording over dataset wording: `perio` → `periodontal disease`
- synonyms allowed: `["dental calculus", "tartar"]`
- category the user cannot explain → flag it, do not invent a term

Render the boxes:

```bash
python $SKILL/scripts/make_preview_nb.py \
    --ann "$DATAS/instances_train.json" --images "$IMAGE_ROOT" \
    --out "$REPORT/step3_preview.ipynb" --n 12
python $SKILL/scripts/run_nb.py "$REPORT/step3_preview.ipynb"
```

`run_nb.py` executes the notebook in place with *this* interpreter. Do not substitute
`jupyter nbconvert`: the CLI is absent from most conda envs that still have nbconvert importable,
a registered `python3` kernelspec can point at an interpreter without matplotlib in it, and a host
Jupyter config (AI Studio's, for one) buries the output in warnings. `run_nb.py` sidesteps all
three. One `WARNING | Kernel is running over TCP without encryption` line is the kernel's own
noise, not a failure.

Writes `$REPORT/step3_preview.png` (annotated samples) and `$REPORT/step3_preview_classes.png`
(boxes per class), plus a geometry check for xyxy-stored-as-xywh and normalised-instead-of-pixel
boxes. Show the user the two PNGs and get explicit confirmation that the boxes sit on the objects.

> **Checkpoint 3** — `check_dataset.py` exited 0; `$DATAS/class_names.json` and
> `$DATAS/class_texts.json` exist and the user approved them; the user confirmed the rendered boxes.
> Report any class with <10 boxes and let the user decide: merge or accept.
>
> Print: `[STEP 3/6] dataset + vocabulary check finished`

## Step 4 — Launchers

Write `$LAUNCH/own_finetune.yaml`. Leave `$LAUNCH/finetune_oraldetect.yaml` alone — it is the
shipped template.

Substitute the expanded value of each shell variable when writing the file:

```yaml
paths:
  wedetect:   $REPO/WeDetect
  config:     $REPO/WeDetect/config/oraldetect_finetune.py
  init:       $WEIGHTS/OralDetect/oraldetect.pth
  text_tower: $WEIGHTS/OralCLIP/oralbert
  work_dir:   $OUT
data:
  data_root:   $IMAGE_ROOT
  train_ann:   $DATAS/instances_train.json
  val_ann:     $DATAS/instances_val.json
  class_names: $DATAS/class_names.json
  class_texts: $DATAS/class_texts.json
  evaluator:   coco
train:
  lr: 5.0e-6
  epochs: 6
  gpus: <N>
  batch_per_gpu: 4
strict_load: true
```

- `evaluator: coco` unless the images carry a `modality` key — `check_dataset.py` reports which.
  `per_modality` raises on datasets built elsewhere.
- from `oraldetect.pth`: `lr 5.0e-6`, `epochs 6`. From `oraldetect_init.pth`: `lr 2.0e-5`,
  `epochs 12`.
- `gpus` must equal `--nproc_per_node`; ask how many the user has. `batch_per_gpu: 4` is ~28 GB at
  1024², use 2 on 24 GB cards.

Then `$LAUNCH/own_eval.yaml`, the same shape with three differences — `checkpoint:` replaces
`init:`, `out_dir:` replaces `work_dir:`, and a bench list is added:

```yaml
paths:
  checkpoint: $WEIGHTS/OralDetect/oraldetect.pth
  out_dir:    $OUT/eval_before
benches:
  - name: val
    ann:  $DATAS/instances_val.json
```

Predictions land at `$OUT/eval_before/val/preds.bbox.json`.

> **Checkpoint 4** — `$LAUNCH/own_finetune.yaml` and `$LAUNCH/own_eval.yaml` exist; every path in
> them passes `test -e`.
>
> Print: `[STEP 4/6] launcher check finished`

## Step 5 — Baseline, dry-run, train

Score the released checkpoint on the val set before changing anything. Without this number there is
no way to say whether the finetune helped.

```bash
python $REPO/run_eval.py --config "$LAUNCH/own_eval.yaml" --dry-run
torchrun --nproc_per_node=<N> $REPO/run_eval.py --config "$LAUNCH/own_eval.yaml"
```

Keep the printed mAP.

```bash
python $REPO/run_finetune.py --config "$LAUNCH/own_finetune.yaml" --dry-run
```

Read the `load_from` line:
- `CLEAN ✅` → proceed.
- any missing / unexpected / shape-mismatch → stop. mmengine drops mismatched keys with only a log
  line, so training would run on a partly random model. Usual causes: >256 classes, or an edited
  model config. Do not set `strict_load: false`.

```bash
torchrun --nproc_per_node=<N> $REPO/run_finetune.py --config "$LAUNCH/own_finetune.yaml"
```

Re-running resumes from `$OUT/last_checkpoint`.

> **Checkpoint 5** — baseline mAP recorded; dry-run printed `CLEAN ✅`; training reached its first
> validation. The dry-run leaves the data pipeline untouched, so the first validation is the first
> proof it works.
>
> Print: `[STEP 5/6] baseline scored; training launched`

## Step 6 — Score the finetune and compare

When training finishes, edit `$LAUNCH/own_eval.yaml` to point at the new checkpoint and a fresh
output directory:

```yaml
paths:
  checkpoint: $OUT/best_coco_bbox_mAP_epoch_<N>.pth
  out_dir:    $OUT/eval_after
```

```bash
torchrun --nproc_per_node=<N> $REPO/run_eval.py --config "$LAUNCH/own_eval.yaml"

python $SKILL/scripts/compare_runs.py --gt "$DATAS/instances_val.json" \
    --before "$OUT/eval_before/val/preds.bbox.json" \
    --after  "$OUT/eval_after/val/preds.bbox.json" \
    --label-before released --label-after finetuned \
    --save "$REPORT/step6_metrics.txt"
```

Reports TP / FP / FN and precision / recall / F1 per class and overall, at a fixed score threshold
and at the best-F1 threshold found by sweeping, plus mAP. There is no accuracy: detection has no
true negatives, so accuracy cannot be computed and anything labelled that way would be invented.

Quote the **best-F1 row together with the threshold that produced it**. P/R/F1 move with the
threshold, and comparing two models at one arbitrary value can invert their ranking.

Then render the same images three ways:

```bash
python $SKILL/scripts/make_compare_nb.py --gt "$DATAS/instances_val.json" --images "$IMAGE_ROOT" \
    --before "$OUT/eval_before/val/preds.bbox.json" \
    --after  "$OUT/eval_after/val/preds.bbox.json" \
    --out "$REPORT/step6_compare.ipynb" --n 8 --score 0.2 \
    --label-before released --label-after finetuned
python $SKILL/scripts/run_nb.py "$REPORT/step6_compare.ipynb"
```

Writes `$REPORT/step6_compare.png`: ground truth / before / after side by side, sorted by how much
the two runs disagree. Show it to the user — it answers what changed, which the metrics cannot.
Typical readings: new boxes on the right that are absent in the middle; boxes that disappeared;
ground truth neither run finds.

> **Checkpoint 6** — `$REPORT/step6_metrics.txt` and `$REPORT/step6_compare.png` exist and both were
> shown to the user.
>
> Print: `[STEP 6/6] before/after compared — all checks finished`

## Final step — The report page

One HTML page, rendered from $SKILL/assets/report_template.html. 

$SKILL/assets/example_report.html is a finished one — open it first and match it: section order, how much copy each section carries, how the numbers read.

```bash
python $SKILL/scripts/make_report_html.py \
    --work-dir "$OUT" \
    --train "$DATAS/instances_train.json" --val "$DATAS/instances_val.json" \
    --title      "在 <dataset folder> 数据集上微调 OralDetect" \
    --subject    "<class name>" --subject-cn "<that class in the reader's language>" \
    --photo-noun "<what the images are>" \
    --gpu "<1×A100-40G>" --minutes <wall clock> \
    --finding    "<one sentence>" \
    --out "$REPORT/report.html"
```

`--title` and `--finding` are the only things it cannot compute. The finding is **one sentence** on what the comparison figure shows that the metrics cannot, from the figure you looked at in step 6, with numbers you can point to.

Add `--standalone --out "$REPORT/report_standalone.html"` for a copy to open or email.

> **Checkpoint ** — `$REPORT/report.html` exists, its numbers match `$REPORT/step6_metrics.txt`, both figures render, the chart draws, and no step claims a tick it cannot evidence. Say in the terminal what the page deliberately leaves out.

> Print: `[FINAL STEP] report page written — all checks finished`
---

## Expected questions

**Detect a class absent from the training data** — add the name to `$DATAS/class_names.json` and a
prompt to `$DATAS/class_texts.json`. No retraining.

**Change image size / text tower / drop calibration** — unsupported from the released checkpoint;
the dry-run refuses. Requires starting from `$WEIGHTS/OralDetect/oraldetect_init.pth`.

**Never modify** `$REPO/WeDetect/config/oraldetect_finetune.py`.

## Hand back

`$REPORT/` — it holds every figure and the metrics table. Plus `$LAUNCH/own_finetune.yaml`,
`$LAUNCH/own_eval.yaml`, `$OUT`, and the command to resume training. If any checkpoint is unmet,
name it and why.
