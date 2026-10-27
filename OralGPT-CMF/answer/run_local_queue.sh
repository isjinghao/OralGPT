#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$ROOT/output/logs"
DATASET="$ROOT/sft/data/turn_level/oralcmf_sft_human_evaluation_rollout.json"
mkdir -p "$LOGS"
exec >>"$LOGS/human_local_queue.log" 2>&1
export PYTHONDONTWRITEBYTECODE=1
ADAPTERS=(none "$ROOT/sft/models/qwen3_5_9b_lora_turn_level" "$ROOT/sft/models/qwen3_5_9b_lora_turn_level_v2" "$ROOT/sft/models/qwen3_5_9b_lora_turn_level_v3")

for version in 0 1 2 3; do
  model="qwen3.5-9b-v$version"
  tokens=2048
  [ "$version" -ne 0 ] || tokens=4096
  extra=()
  if [ "$version" -eq 3 ]; then
    extra=(--question-prompts "$ROOT/sft/data/question_rules/question_prompts.yaml")
  fi
  echo "[$(date -Is)] START $model: 3 shards"
  pids=()
  for shard in 0 1 2; do
    (
      code=0
      bash "$ROOT/answer/run_local.sh" "$model" "${ADAPTERS[$version]}" "$tokens" "$shard" 3 --dataset "$DATASET" "${extra[@]}" >>"$LOGS/human_${model}_shard${shard}.log" 2>&1 || code=$?
      printf "%s\n" "$code" >"$LOGS/human_${model}_shard${shard}.exit_code"
      exit "$code"
    ) &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [ "$failed" -ne 0 ]; then
    echo "[$(date -Is)] FAILED $model; queue stopped"
    echo 1 >"$LOGS/human_local_queue.exit_code"
    exit 1
  fi
  echo "[$(date -Is)] DONE $model"
done
echo 0 >"$LOGS/human_local_queue.exit_code"
echo "[$(date -Is)] DONE all local models"
