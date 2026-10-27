#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/pipeline/treatment_qa.py" \
  --dataset-root "$ROOT/../OralFlow/SH9HCMFdata" \
  --output-dir "$ROOT/output/pipeline/stage4_patient_json_treatment" \
  --overwrite \
  --workers 16 \
  --model gpt-5.4 \
  "$@"
