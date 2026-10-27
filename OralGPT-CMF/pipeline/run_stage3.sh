#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/pipeline/diagnosis_qa.py" \
  --dataset-root "$ROOT/../OralFlow/SH9HCMFdata" \
  --output-dir "$ROOT/output/pipeline/stage3_patient_json_diagnosis_qa" \
  --examination-json-dir "$ROOT/output/pipeline/stage1_patient_json_examination" \
  --xla-ct-json-dir "$ROOT/output/pipeline/stage2_patient_json_XLa_CT" \
  --xray-report-json-dir "$ROOT/output/pipeline/stage2_patient_json_xray_oralagent_report" \
  --model gpt-5.4 \
  --workers 16 \
  --overwrite \
  "$@"
