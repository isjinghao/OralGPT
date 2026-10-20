#!/usr/bin/env bash
# One shard of the strict multi-turn rollout of local Qwen3.5-9B on the 50-patient test split.
# Resumable: rerun the same command after an interruption; finished turns are kept.
# Usage: bash scripts/run_qwen35_rollout.sh <output_model> <adapter_dir|none> <max_new_tokens> <shard_index> <num_shards>
set -euo pipefail

NAME=$1
ADAPTER=$2
MAX_NEW_TOKENS=$3
SHARD_INDEX=$4
NUM_SHARDS=$5
shift 5
ROOT=/root/autodl-tmp/OralGPT/OralGPT-CMF/sft
if [ "$ADAPTER" = none ]; then
  ADAPTER_ARGS=(--no-adapter)
else
  ADAPTER_ARGS=(--adapter "$ADAPTER")
fi

export PATH=/root/autodl-tmp/venvs/qwen35_sft/bin:$PATH
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$ROOT/scripts"

python -u run_sft_test_rollout.py \
  --model /root/autodl-tmp/models/Qwen3.5-9B \
  "${ADAPTER_ARGS[@]}" \
  --template qwen3_5_nothink \
  --llama-root /root/autodl-tmp/LLaMA-Factory-qwen35 \
  --dataset "$ROOT/sft_data/oralcmf_sft_test_rollout.json" \
  --output-root "$ROOT/output" \
  --output-model "$NAME" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --repetition-penalty 1.05 \
  --no-repeat-ngram-size 0 \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  "$@"
