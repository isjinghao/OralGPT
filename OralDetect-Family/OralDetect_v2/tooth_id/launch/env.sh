# Sourced by every launcher. Holds no absolute path: they all come from OralDetect_v2/root_paths.yaml.
_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TOOTHID_ROOT=$(dirname "$_here")
export ORALDETECT_V2_ROOT_PATHS=${ORALDETECT_V2_ROOT_PATHS:-$(dirname "$TOOTHID_ROOT")/root_paths.yaml}
if [ ! -f "$ORALDETECT_V2_ROOT_PATHS" ]; then
    echo "missing $ORALDETECT_V2_ROOT_PATHS -- copy root_paths.example.yaml to it and fill in" >&2
    return 1 2>/dev/null || exit 1
fi
_root() {   # value of a flat `key: value  # comment` line
    awk -v k="$1" -F': *' '$1 == k { sub(/[ \t]*#.*/, "", $2); sub(/[ \t]+$/, "", $2); print $2 }' "$ORALDETECT_V2_ROOT_PATHS"
}
EXP_ROOT=$(_root exp_root); CACHE_ROOT=$(_root cache_root); _pp=$(_root pythonpath)

export TOOTHID_PY=$(_root python)
export TOOTHID_LOGS=$EXP_ROOT/exps_outputs_v2/tooth_id/_slurm_logs
[ -n "$_pp" ] && export PYTHONPATH=$_pp${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
export YOLO_CONFIG_DIR=$CACHE_ROOT/ultralytics       # keep ultralytics' settings off $HOME
export TORCH_HOME=$CACHE_ROOT/torch
export MPLCONFIGDIR=$CACHE_ROOT/matplotlib
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR" "$TOOTHID_LOGS"

DATA=${DATA:-configs/data/iop_compass.yaml}          # relative to the module root
MODEL=${MODEL:-configs/model/yolo11m_seg.yaml}
cd "$TOOTHID_ROOT"
