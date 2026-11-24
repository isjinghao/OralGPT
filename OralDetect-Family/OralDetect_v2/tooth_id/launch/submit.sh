#!/bin/bash
# The way to submit a step. Works from any directory; puts the log under <exp_root>.
#
#   bash launch/submit.sh 01_prepare_data
#   bash launch/submit.sh 02_train
#   EXTRA="--name smoke --set epochs=1 fraction=0.05" bash launch/submit.sh 02_train --time=00:30:00
#   SPLIT=val bash launch/submit.sh 03_evaluate
#
# Anything after the step name goes to sbatch. DATA / MODEL / EXTRA / SPLIT are read by the job.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
step=${1:?usage: submit.sh <01_prepare_data|02_train|03_evaluate|05_zero_shot|07_pseudo_label|08_build_typed_union> [sbatch args]}; shift
export DATA MODEL
sbatch --chdir="$TOOTHID_ROOT" --output="$TOOTHID_LOGS/%x.%j.log" "$@" "launch/$step.slurm"
