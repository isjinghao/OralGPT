#!/usr/bin/env bash
# Launch the local Qwen3.5-9B test rollout as parallel screen sessions (one model copy per shard on GPU 0).
# Resumable: rerun the same command after an interruption.
# Usage: bash scripts/start_qwen35_eval.sh [output_model] [adapter_dir|none] [max_new_tokens] [num_shards]
#   v0 (base):      bash scripts/start_qwen35_eval.sh            (= qwen3.5-9b-v0 none 4096 3)
#   v1 (finetuned): bash scripts/start_qwen35_eval.sh qwen3.5-9b-v1 /root/autodl-tmp/OralGPT/OralGPT-CMF/sft/sft_output/qwen3_5_9b_lora_turn_level 2048
# Logs: output/logs/<session>.log      Attach: screen -r <session>
set -euo pipefail

NAME=${1:-qwen3.5-9b-v0}
ADAPTER=${2:-none}
MAX_NEW_TOKENS=${3:-4096}
NUM_SHARDS=${4:-3}
if [ $# -gt 4 ]; then
  shift 4
else
  set --
fi
ROOT=/root/autodl-tmp/OralGPT/OralGPT-CMF/sft
LOGS="$ROOT/output/logs"
mkdir -p "$LOGS"

for ((i = 0; i < NUM_SHARDS; i++)); do
  session="${NAME//./_}_shard$i"
  if screen -ls | grep -q "[0-9]\.${session}[[:space:]]"; then
    echo "already running: $session"
    continue
  fi
  printf -v command '%q ' bash "$ROOT/scripts/run_qwen35_rollout.sh" "$NAME" "$ADAPTER" "$MAX_NEW_TOKENS" "$i" "$NUM_SHARDS" "$@"
  screen -dmS "$session" bash -o pipefail -c "$command 2>&1 | tee -a '$LOGS/$session.log'"
  echo "started $session -> $LOGS/$session.log"
done
