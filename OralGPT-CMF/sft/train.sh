#!/usr/bin/env bash
# Usage: bash sft/train.sh [config.yaml]
# Resume: set resume_from_checkpoint in the yaml to a checkpoint-XXX directory.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG=${1:-$ROOT/sft/configs/qwen35_lora_v1.yaml}
export PATH=/root/autodl-tmp/venvs/qwen35_sft/bin:$PATH
CONFIG=$(realpath "$CONFIG")
OUT=$(python - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
data = Path(config['dataset_dir'])
info = json.loads((data / 'dataset_info.json').read_text())
names = []
for field in ('dataset', 'eval_dataset'):
    value = config.get(field) or []
    names.extend(value.split(',') if isinstance(value, str) else value)
for name in names:
    target = data / info[name.strip()]['file_name']
    if not target.is_file():
        raise FileNotFoundError(f'Dataset is not ready: {target}. Prepare the configured dataset first.')
print(Path(config['output_dir']).resolve())
PY
)
mkdir -p "$OUT"

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TENSORBOARD_LOGGING_DIR="$OUT/tensorboard"
cd /root/autodl-tmp/LLaMA-Factory-qwen35

llamafactory-cli train "$CONFIG" 2>&1 | tee -a "$OUT/train.log"
