#!/bin/bash
#SBATCH --job-name=encprobe_ft
#SBATCH --cluster=gpu --partition=l40s --account=lzhan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=48g
#SBATCH --time=08:00:00
set -euo pipefail
source launch/env.sh
export HF_HUB_OFFLINE=1
FT=${FT:-configs/probe/tooth_fdi_finetune.yaml}
$ENCPROBE_PY scripts/04_finetune.py --encoder "configs/encoders/${ENCODER:?set ENCODER}.yaml" --probe "$FT" $EXTRA
