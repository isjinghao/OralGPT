#!/bin/bash
# Step 1 (CPU).   bash launch/submit.sh 01_prepare_data
#SBATCH --job-name=toothid_prep
#SBATCH --cluster=htc --account=lzhan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=48g
#SBATCH --time=02:00:00
set -euo pipefail
source launch/env.sh        # cwd is the module root (submit.sh sets --chdir)
$TOOTHID_PY scripts/01_prepare_data.py --data "$DATA" --workers "$SLURM_CPUS_PER_TASK"
