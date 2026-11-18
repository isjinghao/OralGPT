# OralDetect v2

Task-specific detectors that sit beside the open-vocabulary OralDetect (`../OralDetect`).
Each task is a self-contained module with the same skeleton, so a new one is added by copying
the layout, not by reading another module's code.

| module | task | data | status |
|---|---|---|---|
| [`tooth_id/`](tooth_id) | intraoral photo -> every tooth, with its FDI number (box + mask) | IOP-Compass | baseline |
| [`encoder_probe/`](encoder_probe) | frozen vision encoders compared by pretraining (ConvNeXt-B, ViT-B/16) on a tooth-numbering probe | IOP-Compass | first probe |

## Module skeleton

```text
root_paths.yaml                 the ONLY file with absolute paths (git-ignored; copy the .example)
<module>/
├── configs/
│   ├── data/<dataset>.yaml     where the data is, how it is split
│   └── model/<model>.yaml      weights, hyperparameters, where the run goes
├── <package>/                  importable library -- no argparse, no absolute paths
│   ├── data/                   raw reader -> split -> on-disk training format
│   ├── engine/                 train, predict
│   ├── eval/                   metrics
│   └── viz/                    overlays
├── scripts/NN_<step>.py        numbered CLI entry points, run in order; thin
├── launch/                     submit.sh + SLURM launchers + env.sh (reads root_paths.yaml)
└── tests/                      data-free unit tests
```

Rules the skeleton relies on:

1. **No absolute path anywhere but `root_paths.yaml`.** Configs hold paths relative to one of its
   roots (`data_root`, `exp_root`); `<package>/config.py` joins them. A test enforces it.
2. **Configs are the only place with numbers.** Library code takes them as arguments.
3. **One run, one directory**: `<exp_root>/<out_root>/<name>/`, including its resolved config and eval output.
4. **Data and outputs live under the roots**, never in this repo.
5. **Scripts are numbered in execution order** and each is rerunnable on its own.

## Setup on a new machine

```bash
cp root_paths.example.yaml root_paths.yaml      # then fill in the five values
```

`#SBATCH` cluster / partition / account lines in `<module>/launch/*.slurm` are the only other
site-specific text; override them per submission (`bash launch/submit.sh 02_train --partition=...`).
