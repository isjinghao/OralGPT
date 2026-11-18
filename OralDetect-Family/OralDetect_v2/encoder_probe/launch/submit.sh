#!/bin/bash
# bash launch/submit.sh 01_fetch_weights
# ENCODER=vit_b_dinov3 bash launch/submit.sh 02_extract        (or: for e in configs/encoders/*.yaml ...)
# bash launch/submit.sh 03_probe
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
step=${1:?usage: submit.sh <01_fetch_weights|02_extract|03_probe|04_finetune|06_bench_retrieval> [sbatch args]}; shift
export PROBE ENCODER=${ENCODER:-} EXTRA=${EXTRA:-}
sbatch --chdir="$ENCPROBE_ROOT" --output="$ENCPROBE_LOGS/%x.%j.log" "$@" "launch/$step.slurm"
