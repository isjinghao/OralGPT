#!/bin/bash
# Plain launcher -- no job scheduler. Run it anywhere with GPUs visible.
#
#   bash launch_bash/finetune_oraldetect.sh
#
# Fill in finetune_oraldetect.yaml first: the paths in it are placeholders. Everything configurable
# lives there; this file only picks the GPU count and starts torchrun. NPROC must match
# `train.gpus` in the yaml -- run_finetune.py asserts it.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
YAML=$REPO/launch_bash/finetune_oraldetect.yaml
NPROC=${NPROC:-4}

# Optional: keep model/dataset caches off a small home quota.
# export HF_HOME=/path/to/cache/huggingface
# export TORCH_HOME=/path/to/cache/torch

export PYTHONUNBUFFERED=1

# Validate before spending GPUs: checks every path and diffs the checkpoint against the model.
python "$REPO/run_finetune.py" --config "$YAML" --dry-run

torchrun --nproc_per_node="$NPROC" "$REPO/run_finetune.py" --config "$YAML"
