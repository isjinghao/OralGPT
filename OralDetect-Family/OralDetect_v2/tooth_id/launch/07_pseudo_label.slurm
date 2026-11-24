#!/bin/bash
# Step 7: number the teeth of unlabeled image sets with a trained run.
#   PL_DATA=configs/data_unlabeled/intraoral_detection.yaml EXTRA="--sheet-dir <folder>" bash launch/submit.sh 07_pseudo_label
#SBATCH --job-name=toothid_pseudo
#SBATCH --cluster=gpu --partition=l40s --account=lzhan
#SBATCH --exclude=gpu-n60,gpu-n73
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=32g
#SBATCH --time=03:00:00
set -euo pipefail
source launch/env.sh        # cwd is the module root (submit.sh sets --chdir)
PL_DATA=${PL_DATA:-configs/data_unlabeled/intraoral_detection.yaml}; EXTRA=${EXTRA:-}
$TOOTHID_PY scripts/07_pseudo_label.py --data "$PL_DATA" --model "$MODEL" $EXTRA
