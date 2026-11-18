# tooth_id — intraoral FDI tooth numbering

Input: one intraoral photograph. Output: every visible tooth as a box + mask, labelled with its
FDI number (32 classes, permanent dentition 11–48).

Baseline: YOLO11m box detection at 1024 px, trained on IOP-Compass (`configs/model/yolo11m_det.yaml`,
the default). `yolo11m_seg.yaml` is the box + mask variant: `MODEL=configs/model/yolo11m_seg.yaml`.

## Layout

```text
tooth_id/
├── configs/
│   ├── data/iop_compass.yaml       raw + processed paths, patient-level split, image size
│   └── model/yolo11m_{det,seg}.yaml  weights, train / predict / eval settings, output root
├── toothid/
│   ├── fdi.py                      THE label space: 32 codes, mirror(), error taxonomy
│   ├── config.py                   yaml loading, run_dir()
│   ├── data/
│   │   ├── iop_compass.py          raw reader (Patient_*/IOP_<View>_*.{png,json})
│   │   ├── split.py                patient-level split
│   │   └── yolo_format.py          YOLO-seg writer / label reader
│   ├── engine/
│   │   ├── train.py                ultralytics wrapper; refuses flip augmentation
│   │   └── predict.py              split -> prediction records
│   ├── eval/
│   │   ├── matching.py             class-agnostic IoU matching
│   │   └── id_metrics.py           det_recall / id_accuracy / e2e_recall, error breakdown
│   └── viz/overlay.py              GT-vs-prediction overlay
├── scripts/
│   ├── 01_prepare_data.py          raw -> YOLO-seg dataset
│   ├── 02_train.py                 train
│   ├── 03_evaluate.py              mAP + tooth-ID metrics on a split
│   └── 04_visualize.py             overlays (random or worst-first)
├── launch/
│   ├── submit.sh                   THE way to submit a step (any cwd; log under <exp_root>)
│   ├── env.sh                      reads ../root_paths.yaml: interpreter, caches, log dir
│   ├── 01_prepare_data.slurm       htc, CPU
│   ├── 02_train.slurm              gpu, 1 GPU; runs 02 -> 03 -> 04
│   └── 03_evaluate.slurm           gpu, 1 GPU; runs 03 -> 04
└── tests/test_core.py              label space, split, metrics — no data needed
```

## Run

```bash
cp ../root_paths.example.yaml ../root_paths.yaml   # first time only: fill in local roots
bash launch/submit.sh 01_prepare_data   # once, ~10 min
bash launch/submit.sh 02_train          # train + test-set eval + overlays

# smoke test: 1 epoch on 5% of train, separate run dir
EXTRA="--name smoke --set epochs=1 fraction=0.05" bash launch/submit.sh 02_train --time=00:30:00
# another split / another run
SPLIT=val EXTRA="--name smoke" bash launch/submit.sh 03_evaluate
```

## Where things land

`<data_root>` and `<exp_root>` come from `../root_paths.yaml`; the rest from the two configs.

| what | where |
|---|---|
| raw data | `<data_root>/<raw_root>` |
| YOLO dataset (`images/ labels/ dataset.yaml manifest.jsonl splits.json`) | `<data_root>/<yolo_root>` |
| a run (`weights/ results.csv resolved_config.yaml`) | `<exp_root>/<out_root>/<name>/` |
| evaluation (`metrics.json report.md predictions.jsonl viz/`) | `<exp_root>/<out_root>/<name>/eval_<split>/` |
| SLURM logs | `<exp_root>/<out_root>/_slurm_logs/` |

## Data: IOP-Compass

1,000 patients × 5 views (Center / Upper / Down / Left / Right) = 5,000 photos, 88,601 teeth.
Split 800 / 100 / 100 **patients** (seed 0). Images are stored as JPEGs with the long side capped
at 1280 px; labels are normalised polygons, so they are resolution-independent.

Known properties of the labels:

- Permanent codes only (11–48). No primary-tooth codes exist in the release.
- Third molars are rare: about 200–280 instances each, against 2,400–3,950 for every other code.
- 33 teeth are annotated as several contours; the largest is kept.

## Metrics

mAP alone cannot say whether a tooth was missed or found and misnumbered, so step 3 reports both.

| metric | definition |
|---|---|
| box / mask mAP | ultralytics `val`, class-aware |
| `det_recall` | found teeth / GT teeth, number ignored (IoU ≥ 0.5, one-to-one) |
| `det_precision` | found teeth / predicted teeth |
| `id_accuracy` | correctly numbered / found teeth |
| `e2e_recall` | found and correctly numbered / GT teeth |

All four are also broken down per view and per FDI code. Wrong numbers are classified by
`fdi.relation()`: `adjacent`, `same_quadrant`, `mirror`, `same_arch`, `other_arch`.

## Two constraints that are easy to break

- **No flips.** A horizontal flip turns tooth 16 into 26 while the label still says 16.
  `engine/train.py` raises unless `fliplr` and `flipud` are both 0.
- **Split by patient.** Use `data/split.py`; never split the image list.

## Adding a model or a dataset

- New model: add `configs/model/<name>.yaml`, pass it with `MODEL=configs/model/<name>.yaml bash launch/submit.sh 02_train`.
  Any ultralytics detect or seg checkpoint works unchanged (mask mAP is reported only for seg).
- New dataset: add a reader next to `data/iop_compass.py` yielding `Record` / `Tooth`, a
  `configs/data/<name>.yaml`, and reuse `yolo_format.py` as is.
