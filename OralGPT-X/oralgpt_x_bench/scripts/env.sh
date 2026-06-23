#!/usr/bin/env bash
# Shared environment for OralGPT-X-Bench scripts.
set -euo pipefail

BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${model_path:?Set model_path to your BAGEL checkpoint directory}"
: "${output_path:?Set output_path for evaluation artifacts}"

if [[ -z "${BAGEL_ROOT:-}" ]]; then
  if [[ -d "${BENCH_ROOT}/../../../OralGPT-X/Bagel" ]]; then
    export BAGEL_ROOT="$(cd "${BENCH_ROOT}/../../../OralGPT-X/Bagel" && pwd)"
  elif [[ -d "${BENCH_ROOT}/../Bagel" ]]; then
    export BAGEL_ROOT="$(cd "${BENCH_ROOT}/../Bagel" && pwd)"
  else
    echo "Set BAGEL_ROOT to your Bagel repository root." >&2
    exit 1
  fi
fi

if [[ ! -f "${BAGEL_ROOT}/modeling/bagel/bagel.py" ]]; then
  echo "Invalid BAGEL_ROOT=${BAGEL_ROOT}" >&2
  exit 1
fi

export GPUS="${GPUS:-8}"

mkdir -p "${output_path}"

echo "BENCH_ROOT=${BENCH_ROOT}"
echo "BAGEL_ROOT=${BAGEL_ROOT}"
echo "model_path=${model_path}"
echo "output_path=${output_path}"
