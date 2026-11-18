#!/bin/bash
#SBATCH --job-name=encprobe_probe
#SBATCH --cluster=gpu --partition=l40s --account=lzhan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=48g
#SBATCH --time=02:00:00
set -euo pipefail
source launch/env.sh
$ENCPROBE_PY scripts/03_probe.py --probe "$PROBE"
