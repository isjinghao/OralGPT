#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if ! command -v conda >/dev/null; then
  for CONDA_HOME in "$HOME/miniforge3" "$HOME/miniconda3"; do
    [[ -x "$CONDA_HOME/bin/conda" ]] && export PATH="$CONDA_HOME/bin:$PATH" && break
  done
fi
eval "$(conda shell.bash hook)"
conda activate cmfbench
export FASTEMBED_CACHE_PATH="${FASTEMBED_CACHE_PATH:-/root/autodl-tmp/fastembed_cache}"
export MEM0_TELEMETRY="${MEM0_TELEMETRY:-False}"
export MEMO_OPENAI_BASE_URL="${MEMO_OPENAI_BASE_URL:-http://127.0.0.1:8004/v1}"
export MEMO_OPENAI_MODEL="${MEMO_OPENAI_MODEL:-qwen2.5-7b-instruct-memory}"
export EMBEDDING_OPENAI_BASE_URL="${EMBEDDING_OPENAI_BASE_URL:-http://127.0.0.1:8005/v1}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-qwen3-embedding-0.6b}"
python -u -m report_pipeline.run_step4_report \
  --phase answers \
  --answer-workers 2 \
  --method-workers 1 \
  "$@"
