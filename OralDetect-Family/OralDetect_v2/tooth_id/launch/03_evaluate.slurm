#!/bin/bash
# Step 3 + 4 alone, for a run that already has weights.
#   SPLIT=val bash launch/submit.sh 03_evaluate
#SBATCH --job-name=toothid_eval
#SBATCH --cluster=gpu --partition=a100 --account=lzhan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=32g
#SBATCH --time=01:00:00
set -euo pipefail
source launch/env.sh        # cwd is the module root (submit.sh sets --chdir)
SPLIT=${SPLIT:-test}; EXTRA=${EXTRA:-}
$TOOTHID_PY scripts/03_evaluate.py  --data "$DATA" --model "$MODEL" --split "$SPLIT" $EXTRA
$TOOTHID_PY scripts/04_visualize.py --data "$DATA" --model "$MODEL" --split "$SPLIT" --n 20 --worst $EXTRA
