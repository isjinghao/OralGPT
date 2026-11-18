#!/bin/bash
#SBATCH --job-name=encprobe_retr
#SBATCH --cluster=gpu --partition=l40s --account=lzhan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=48g
#SBATCH --time=01:00:00
set -euo pipefail
source launch/env.sh
export HF_HUB_OFFLINE=1
$ENCPROBE_PY scripts/06_bench_retrieval.py $EXTRA
