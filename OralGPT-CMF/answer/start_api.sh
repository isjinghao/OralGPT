#!/usr/bin/env bash
# Launch answer generation for gpt-5.5, gemini-3.5-flash and qwen3.8-flash on the 50 test patients,
# one screen session per model. Resumable: rerun the same command after an interruption.
# Keys/URLs come from .env (ANSWER_/GEMINI_/QWEN_OPENAI_API_KEY and _BASE_URL).
# Usage: [MODELS="gemini-3.5-flash"] bash answer/start_api.sh [extra generate_api.py args, e.g. --limit 2]
# Logs:  output/logs/<model>.log      Attach: screen -r api_gpt-5_5 / api_gemini-3_5-flash / api_qwen3_8-flash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$ROOT/output/logs"
mkdir -p "$LOGS"
set -a
source "$ROOT/.env"
set +a
MODELS=${MODELS:-"gpt-5.5 gemini-3.5-flash qwen3.8-flash"}
EXTRA=""
if [ $# -gt 0 ]; then
  EXTRA=$(printf '%q ' "$@")
fi

start() {
  local model=$1 key_env=$2 base_url=$3 workers=$4 transport=${5:-}
  [[ " $MODELS " == *" $model "* ]] || return 0
  local session="api_${model//./_}"
  if screen -ls | grep -q "[0-9]\.${session}[[:space:]]"; then
    echo "already running: $session"
    return
  fi
  screen -dmS "$session" bash -o pipefail -c "bash '$ROOT/answer/run_api.sh' --answer-model '$model' --api-key-env '$key_env' --answer-base-url '$base_url' --num-workers $workers $transport $EXTRA 2>&1 | tee -a '$LOGS/$model.log'"
  echo "started $session (workers=$workers $transport) -> $LOGS/$model.log"
}

start gpt-5.5 ANSWER_OPENAI_API_KEY "$ANSWER_OPENAI_BASE_URL" 12
start gemini-3.5-flash GEMINI_OPENAI_API_KEY "$GEMINI_OPENAI_BASE_URL" 1 "--stream --timeout 180"
start qwen3.8-flash QWEN_OPENAI_API_KEY "$QWEN_OPENAI_BASE_URL" 12 "--timeout 300"
