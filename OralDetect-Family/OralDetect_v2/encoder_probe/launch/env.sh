# Sourced by every launcher. Holds no absolute path: they all come from OralDetect_v2/root_paths.yaml.
_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ENCPROBE_ROOT=$(dirname "$_here")
export ORALDETECT_V2_ROOT_PATHS=${ORALDETECT_V2_ROOT_PATHS:-$(dirname "$ENCPROBE_ROOT")/root_paths.yaml}
_root() { awk -v k="$1" -F': *' '$1 == k { sub(/[ \t]*#.*/, "", $2); sub(/[ \t]+$/, "", $2); print $2 }' "$ORALDETECT_V2_ROOT_PATHS"; }
EXP_ROOT=$(_root exp_root); CACHE_ROOT=$(_root cache_root)
export ENCPROBE_PY=$(_root python)
export ENCPROBE_LOGS=$EXP_ROOT/exps_outputs_v2/encoder_probe/_slurm_logs
export PYTHONUNBUFFERED=1 HF_HOME=$CACHE_ROOT/hf TORCH_HOME=$CACHE_ROOT/torch
mkdir -p "$HF_HOME" "$ENCPROBE_LOGS"
PROBE=${PROBE:-configs/probe/tooth_fdi.yaml}
cd "$ENCPROBE_ROOT"
