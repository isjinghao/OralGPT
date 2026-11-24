#!/bin/bash
# Step 8 (CPU).   DATA=configs/data/iop_plus_io150k.yaml bash launch/submit.sh 08_build_typed_union
#SBATCH --job-name=toothid_union
#SBATCH --cluster=htc --account=lzhan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32g
#SBATCH --time=01:00:00
set -euo pipefail
source launch/env.sh        # cwd is the module root (submit.sh sets --chdir)
$TOOTHID_PY tests/test_core.py
$TOOTHID_PY scripts/08_build_typed_union.py --data "$DATA"
