#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/pipeline/xray_report.py" \
  --dataset-root "$ROOT/../OralFlow/SH9HCMFdata" \
  --output-dir "$ROOT/output/pipeline/stage2_patient_json_xray_oralagent_report" \
  --api-base http://0.0.0.0:8124/v1 \
  --model local-agent \
  --workers 1 \
  --overwrite \
  "$@"
