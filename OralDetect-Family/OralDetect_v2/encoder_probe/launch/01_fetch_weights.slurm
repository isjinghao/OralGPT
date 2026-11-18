#!/bin/bash
#SBATCH --job-name=encprobe_fetch
#SBATCH --cluster=htc --account=lzhan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48g
#SBATCH --time=01:00:00
set -euo pipefail
source launch/env.sh
$ENCPROBE_PY scripts/01_fetch_weights.py $EXTRA
