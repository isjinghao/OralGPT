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
python -u -m report_pipeline.step1_report_trajectory.run_perception_trajectory "$@"
