#!/usr/bin/env bash
# T2I benchmark: infer -> GPT judge -> summarize
set -euo pipefail

source "$(dirname "$0")/env.sh"

SUBSET=t2i
PRED_DIR="${output_path}/inference/${SUBSET}"
METRICS_JSONL="${output_path}/metrics/judge_${SUBSET}.jsonl"
SUMMARY_DIR="${output_path}/summary"

META="${BENCH_ROOT}/benchmark/${SUBSET}/metadata.examples.json"
if [[ "${FULL_EVAL:-0}" == "1" && -f "${BENCH_ROOT}/benchmark/${SUBSET}/metadata.test.json" ]]; then
  META="${BENCH_ROOT}/benchmark/${SUBSET}/metadata.test.json"
elif [[ "${FULL_EVAL:-0}" == "1" && -f "${BENCH_ROOT}/benchmark/${SUBSET}/metadata.test.jsonl" ]]; then
  META="${BENCH_ROOT}/benchmark/${SUBSET}/metadata.test.jsonl"
fi

if [[ ! -f "${META}" ]]; then
  echo "Missing metadata: ${META}" >&2
  exit 1
fi

echo "Using metadata: ${META}"

echo "=== Stage 1: T2I inference ==="
cd "${BAGEL_ROOT}"
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="${GPUS}" \
  --master_addr=127.0.0.1 \
  --master_port=12358 \
  "${BENCH_ROOT}/infer/gen_t2i_mp.py" \
  --metadata_file "${META}" \
  --output_dir "${PRED_DIR}" \
  --model-path "${model_path}" \
  --bagel-root "${BAGEL_ROOT}" \
  --resolution "${T2I_RESOLUTION:-512}"

echo "=== Stage 2: GPT judge ==="
python "${BENCH_ROOT}/metrics/judge_t2i.py" \
  --metadata_file "${META}" \
  --pred_dir "${PRED_DIR}" \
  --output_jsonl "${METRICS_JSONL}" \
  --benchmark "${SUBSET}" \
  --judge_mode "${JUDGE_MODE:-auto}"

echo "=== Stage 3: summarize ==="
cd "${BENCH_ROOT}"
python -m summarize.summarize \
  --benchmark "${SUBSET}" \
  --metrics_jsonl "${METRICS_JSONL}" \
  --output_dir "${SUMMARY_DIR}"

echo "T2I evaluation complete. Summary: ${SUMMARY_DIR}/${SUBSET}_summary.json"
