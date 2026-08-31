# OralDetect

Open-vocabulary dental object detection.

The class name is the classifier, the model can detect the class that was never trained on, just add the name to the vocabulary file.

## Layout

```text
OralDetect-Family/
├── run_finetune.py                 train / finetune entry
├── run_eval.py                     eval entry
└── OralDetect/
    ├── launch_bash/                edit the .yaml; run a .sh or .slurm
    │   ├── {finetune,eval}_oraldetect.yaml     ← the only file you edit
    │   ├── {finetune,eval}_oraldetect.sh          plain torchrun
    │   └── {finetune,eval}_oraldetect.slurm       SLURM
    ├── datas/                      COCO annotations + 87-class vocabulary
    ├── bench_jsons/schemeB/        the 5 per-modality test benches
    ├── tools/per_modality_eval.py  offline scoring
    └── WeDetect/                   framework, one config: oraldetect_finetune.py
```

## 1. Env

CUDA 12.x, one or more GPUs.

```bash
conda create -n oraldetect python=3.10 -y && conda activate oraldetect

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install "setuptools<70" wheel "numpy<2"
MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" \
    pip install mmcv==2.1.0 --no-binary mmcv --no-build-isolation
pip install mmengine==0.10.7 mmdet==3.3.0 timm==0.9.16 \
            albumentations==1.3.1 opencv-python-headless transformers pyyaml
```

Three pins are **not optional**:

| pin | why |
|---|---|
| `numpy<2` | mmengine silently upgrades it, then the data pipeline breaks |
| `albumentations==1.3.1` | 1.4+ rejects the extra keys `mmdet.Albu` passes |
| `opencv-python-headless` only | never alongside `opencv-python` — both own `cv2/` |

## 2. Data and weights


| | |
|---|---|
| Weights | [OralGPT/OralDetect-Family](https://huggingface.co/OralGPT/OralDetect-Family) — **required** |
| Our training data | [OralGPT/OralDetect-Training](https://huggingface.co/datasets/OralGPT/OralDetect-Training) — optional |
| Our benchmark | [OralGPT/OralDetect-Bench](https://huggingface.co/datasets/OralGPT/OralDetect-Bench) — optional |

```bash
export DATA=/path/to/oraldetect            # wherever you keep datasets

# weights (2.7 G)
hf download OralGPT/OralDetect-Family --local-dir $DATA/weights

# our training set — only if you want to reproduce or keep training on it
hf download OralGPT/OralDetect-Training --repo-type dataset \
    --include "Detect-Training/*" --local-dir $DATA/dl
mkdir -p $DATA/images
for t in $DATA/dl/Detect-Training/*.tar; do tar -xf "$t" -C $DATA/images/; done
cp $DATA/dl/Detect-Training/*.json datas/
```

What the weights repo contains:

| file | size | what |
|---|--:|---|
| `OralDetect/oraldetect.pth` | 0.83 G | **the detector** — start here for inference or finetuning |
| `OralDetect/oraldetect_init.pth` | 0.41 G | training init: towers loaded, neck/head still random |
| `OralDetect/class_{names,texts}_oraldetect.json` | 6 K | the 87-class vocabulary and its prompts |
| `OralCLIP/oralbert/` | 0.41 G | the dental BERT text tower, HF format — **required at run time** |
| `OralCLIP/oralclip.pt` | 0.75 G | the full dental CLIP (`vision.*` + `text.*`) |
| `OralCLIP/vision_tower.pt` | 0.34 G | its vision tower alone |


Our images extract as `<modality>/<name>`, matching the `file_name` in our COCO files. For your own dataset, skip to §3.


<details><summary>Our dataset sizes</summary>

| split | images | boxes | intraoral | panoramic | periapical | cytology | histology |
|---|--:|--:|--:|--:|--:|--:|--:|
| train | 33,816 | 813,729 | 6,874 | 15,718 | 5,038 | 5,870 | 316 |
| val | 4,672 | 115,779 | 969 | 2,260 | 560 | 848 | 35 |
| test | 3,400 | 57,946 | 900 | 1,000 | 600 | 800 | 100 |

</details>

## 3. Prepare your dataset

Three files. Standard COCO detection format plus two vocabulary files.

```text
your_data/
├── images/                     `file_name` is relative to this
├── instances_train.json        COCO
├── instances_val.json          COCO
├── class_names.json            list[str]        -> ["caries", "calculus", ...]
└── class_texts.json            list[list[str]]  -> [["caries"], ["dental calculus", "tartar"]]
```

🔴 The loader matches categories **by name**, not by
id, `cat_ids = coco.get_cat_ids(cat_names=class_names)`, then `label = position in class_names`.
So every `categories[].name` in your COCO json must be a **byte-exact** match for a string in `class_names.json`. A name that does not match is dropped without an error, and you train on empty annotations. `categories[].id` can be any integers you like.

`class_texts.json` is in the **same order** as `class_names.json`, one list of prompts per class.
These strings are what the text tower embeds, and that embedding is the class prototype, so they matter more than a label name usually does.
Prefer what a clinician would write (`"dental calculus"`) over a dataset code (`"DC"`).

Limits: **≤ 256 classes**, and boxes in `xywh` like COCO style.

## 4. Finetune

Edit `launch_bash/finetune_oraldetect.yaml` — point it at your data. Nothing else
changes.

```yaml
paths:
  wedetect:   WeDetect
  config:     WeDetect/config/oraldetect_finetune.py
  init:       $DATA/weights/OralDetect/oraldetect.pth   # the released detector
  text_tower: $DATA/weights/OralCLIP/oralbert           # HF-format directory
  work_dir:   /path/to/output
data:
  data_root:   your_data/images/
  train_ann:   your_data/instances_train.json
  val_ann:     your_data/instances_val.json
  class_names: your_data/class_names.json
  class_texts: your_data/class_texts.json
  evaluator:   coco            # required for any dataset that is not ours -- see below
train:
  lr: 5.0e-6                   # from the released detector; use 2.0e-5 from oraldetect_init.pth
  epochs: 6
  gpus: 4
```

recommend to validate first without GPU, one minute:

```bash
python run_finetune.py --config launch_bash/finetune_oraldetect.yaml --dry-run
```

It checks every path, resolves the config, and the checkpoint of the model.
`load_from` drops mismatched keys with only a log line. It refuses to start on any mismatch.

Then:

```bash
torchrun --nproc_per_node=4 run_finetune.py \
    --config launch_bash/finetune_oraldetect.yaml
```

`--nproc_per_node` must match `train.gpus`. Checkpoints go to `paths.work_dir`; re-running resumes
from the last one.

Two ready-made launchers wrap exactly the two commands above:

| | |
|---|---|
| `finetune_oraldetect.sh` | plain `torchrun`, no scheduler — runs the dry-run, then the job |
| `finetune_oraldetect.slurm` | SLURM; every `#SBATCH` line is a placeholder to edit |

**Two things to know**

- **`evaluator: coco` is required.**
- **≤ 256 classes is shape-safe.** The head sizes its classifier as `max(256, num_classes)` and class
  prototypes come from the text tower rather than a learned per-class matrix. 

Adding a class the model was never trained on is a one-line edit to the two vocabulary files.

## 5. Test

Same shape, in `eval_oraldetect.yaml`:

```yaml
paths:
  checkpoint: /path/to/output/best_coco_bbox_mAP_epoch_N.pth   # or the released oraldetect.pth
  out_dir:    /path/to/eval_output
data:
  data_root:   your_data/images/
  class_names: your_data/class_names.json
  class_texts: your_data/class_texts.json
  evaluator:   coco
benches:
  - name: test
    ann:  your_data/instances_test.json
```

```bash
python run_eval.py --config launch_bash/eval_oraldetect.yaml --dry-run
torchrun --nproc_per_node=4 run_eval.py --config launch_bash/eval_oraldetect.yaml
```

Every entry under `benches` is scored in one run, each writing its own `preds.bbox.json` under
`out_dir/<name>/`. `--bench <name>` runs just one.
`eval_oraldetect.sh` and `eval_oraldetect.slurm` wrap these two commands.

With `evaluator: coco` you get standard COCO mAP. Re-score any prediction file offline:

```bash
python tools/per_modality_eval.py your_data/instances_test.json <run>/preds.bbox.json
```
