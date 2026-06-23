#!/usr/bin/env bash
# MRI benchmark: infer -> pixel metrics -> summarize
set -euo pipefail

source "$(dirname "$0")/env.sh"

SUBSET=mri
PRED_DIR="${output_path}/inference/${SUBSET}"
METRICS_JSONL="${output_path}/metrics/pixel_${SUBSET}.jsonl"
SUMMARY_DIR="${output_path}/summary"

# Default: 5 public examples in repo. Set FULL_EVAL=1 to export full test split locally.
if [[ "${FULL_EVAL:-0}" == "1" ]]; then
  META="${BENCH_ROOT}/benchmark/${SUBSET}/metadata.test.json"
  bench_data_root="${bench_data_root:-${BENCH_ROOT}/benchmark_data/mri}"
  PARQUET_ROOT="${parquet_root:-/data/OralGPT/OralGPT-X/dataset_MRI_T1_T2/test}"
  if [[ ! -f "${META}" ]]; then
    echo "Exporting full MRI test split from ${PARQUET_ROOT}..."
    python "${BENCH_ROOT}/tools/export_mri_benchmark.py" \
      --parquet-root "${PARQUET_ROOT}" \
      --bench-data-root "${bench_data_root}" \
      --output-metadata "${META}"
  fi
else
  META="${BENCH_ROOT}/benchmark/${SUBSET}/metadata.examples.json"
  bench_data_root="${bench_data_root:-${BENCH_ROOT}/benchmark/${SUBSET}/examples}"
fi

export bench_data_root

if [[ ! -f "${META}" ]]; then
  echo "Missing metadata: ${META}" >&2
  exit 1
fi

echo "Using metadata: ${META}"
echo "bench_data_root: ${bench_data_root}"

echo "=== Stage 1: inference ==="
cd "${BAGEL_ROOT}"
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="${GPUS}" \
  --master_addr=127.0.0.1 \
  --master_port=12356 \
  "${BENCH_ROOT}/infer/gen_edit_mp.py" \
  --metadata_file "${META}" \
  --bench_data_root "${bench_data_root}" \
  --output_dir "${PRED_DIR}" \
  --model-path "${model_path}" \
  --bagel-root "${BAGEL_ROOT}"

echo "=== Stage 2: pixel metrics ==="
python "${BENCH_ROOT}/metrics/pixel.py" \
  --metadata_file "${META}" \
  --bench_data_root "${bench_data_root}" \
  --pred_dir "${PRED_DIR}" \
  --output_jsonl "${METRICS_JSONL}" \
  --benchmark "${SUBSET}"

echo "=== Stage 3: summarize ==="
cd "${BENCH_ROOT}"
python -m summarize.summarize \
  --benchmark "${SUBSET}" \
  --metrics_jsonl "${METRICS_JSONL}" \
  --output_dir "${SUMMARY_DIR}"

echo "MRI evaluation complete. Summary: ${SUMMARY_DIR}/${SUBSET}_summary.json"
