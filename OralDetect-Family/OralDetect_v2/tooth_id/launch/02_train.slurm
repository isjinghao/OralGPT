#!/bin/bash
# Step 2 (1 GPU), then step 3 + 4 on the test split.
#   bash launch/submit.sh 02_train
#   EXTRA="--name smoke --set epochs=1 fraction=0.05" bash launch/submit.sh 02_train --time=00:30:00
#SBATCH --job-name=toothid_train
#SBATCH --cluster=gpu --partition=l40s --account=lzhan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:1 --mem=64g
#SBATCH --time=12:00:00
set -euo pipefail
source launch/env.sh        # cwd is the module root (submit.sh sets --chdir)
EXTRA=${EXTRA:-}
NAME_ARG=$(echo "$EXTRA" | grep -oE -- '--name +[^ ]+' || true)       # eval must look at the same run

$TOOTHID_PY scripts/02_train.py    --data "$DATA" --model "$MODEL" $EXTRA
$TOOTHID_PY scripts/03_evaluate.py --data "$DATA" --model "$MODEL" --split test $NAME_ARG
$TOOTHID_PY scripts/04_visualize.py --data "$DATA" --model "$MODEL" --split test --n 20 --worst $NAME_ARG
