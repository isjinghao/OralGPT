#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/pipeline/face_eye_mosaic.py" \
  --root "$ROOT/../OralFlow/SH9HCMFdata" \
  --min-detection-confidence 0.2 \
  --detect-short-side 960 \
  --overwrite \
  "$@"
