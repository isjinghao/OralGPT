#!/usr/bin/env bash
# Show test-rollout progress for every model under output/.
# Usage: bash scripts/eval_progress.sh
ROOT=/root/autodl-tmp/OralGPT/OralGPT-CMF/sft
LOGS="$ROOT/output/logs"

screen -ls | grep -E 'api_|_shard' || echo "no eval screen sessions"
for log in "$LOGS"/*.log; do
  [ -e "$log" ] || continue
  echo "== $(basename "$log" .log)  done_turns=$(grep -cE 'api finish|\]\[turns [0-9]+/' "$log")  problems=$(grep -ciE 'Traceback|FAILED|retry|empty answer|out of memory' "$log")"
  grep -E 'progress|\]\[turns |batch\]\[done|FAILED|Error' "$log" | tail -n 1 | cut -c1-240
done
echo "== answers.json per model"
find "$ROOT/output" -path '*/evaluation/*/answers.json' | awk -F/ '{print $(NF-1)}' | sort | uniq -c
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
