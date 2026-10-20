#!/usr/bin/env bash
# Usage: bash api_eval/run_answers.sh [--num-workers 4] [--limit N] [--case-id ID] [--answer-model gpt-5] ...
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate cmfbench
python -u api_eval/run_answers.py "$@"
