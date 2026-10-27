#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/pipeline/xla_ct.py" \
  --dataset-root "$ROOT/../OralFlow/SH9HCMFdata" \
  --output-dir "$ROOT/output/pipeline/stage2_patient_json_XLa_CT" \
  --model gpt-5 \
  --workers 16 \
  --overwrite \
  "$@"
