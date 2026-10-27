#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH=/root/autodl-tmp/venvs/qwen35_sft/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
RUN=all77
failed=0
for model in gpt-5.5 gemini-3.5-flash qwen3.8-flash qwen3.5-9b-v0 qwen3.5-9b-v1 qwen3.5-9b-v3; do
  echo "[$(date -Is)] START $model: judge=gpt-5 workers=16"
  code=0
  python -u scoring/score_answers.py \
    --answer-model "$model" --judge-model gpt-5 --run-name "$RUN" \
    --num-workers 16 --max-tokens 16384 --max-attempts 3 \
    --timeout 300 --reasoning-effort low --allow-http || code=$?
  echo "[$(date -Is)] DONE $model exit=$code"
  if [ "$code" -ne 0 ]; then failed=1; fi
done
echo "[$(date -Is)] DONE all baselines exit=$failed"
exit "$failed"
