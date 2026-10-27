#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/pipeline/examination_qa.py" \
  --dataset-root "$ROOT/../OralFlow/SH9HCMFdata" \
  --output-dir "$ROOT/output/pipeline/stage1_patient_json_examination" \
  --overwrite \
  --workers 16 \
  --model gpt-5.4 \
  "$@"
