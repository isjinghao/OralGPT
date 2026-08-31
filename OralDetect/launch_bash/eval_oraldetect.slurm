#!/bin/bash
# SLURM launcher. Every #SBATCH line below is a placeholder -- partition, account and the resource
# limits will not work anywhere unedited.
#
#   sbatch launch_bash/eval_oraldetect.slurm
#
#SBATCH --job-name=oraldetect_eval
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:4 --mem=180g
#SBATCH --time=04:00:00
#SBATCH --output=oraldetect_eval.%j.log
set -euo pipefail

REPO=/path/to/OralDetect-Family
YAML=$REPO/launch_bash/eval_oraldetect.yaml
NPROC=4                                   # must match --gres=gpu:N above and `eval.gpus` in the yaml

# Activate however your site does it -- conda, micromamba, module load, a venv.
# conda activate oraldetect

export PYTHONUNBUFFERED=1
# export HF_HOME=/path/to/cache/huggingface
# export TORCH_HOME=/path/to/cache/torch

torchrun --nproc_per_node=$NPROC --master_port=$((29500 + RANDOM % 1000)) \
    "$REPO/run_eval.py" --config "$YAML"
