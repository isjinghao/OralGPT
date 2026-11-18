#!/bin/bash
# Step 5: zero-shot test on an external tooth-type set.
#   ZS_DATA=configs/data_external/io150k.yaml bash launch/submit.sh 05_zero_shot
#SBATCH --job-name=toothid_zeroshot
#SBATCH --cluster=gpu --partition=l40s --account=lzhan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=32g
#SBATCH --time=01:00:00
set -euo pipefail
source launch/env.sh        # cwd is the module root (submit.sh sets --chdir)
ZS_DATA=${ZS_DATA:-configs/data_external/io150k.yaml}; EXTRA=${EXTRA:-}
$TOOTHID_PY scripts/05_zero_shot_types.py --data "$ZS_DATA" --model "$MODEL" $EXTRA
