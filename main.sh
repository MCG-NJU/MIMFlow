#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <fit|validate|test|predict> <config.yaml> [extra LightningCLI args...]" >&2
  echo "Example: $0 fit configs/mimflow_l_phase1.yaml --data.data_root /path/to/imagenet" >&2
  exit 1
}

if [ $# -lt 2 ]; then
  usage
fi

TASK=$1
CONFIG=$2
shift 2

NPROC_PER_NODE=${NPROC_PER_NODE:-1}

if [ "${NPROC_PER_NODE}" = "1" ]; then
  python3 main.py "${TASK}" -c "${CONFIG}" "$@"
else
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" main.py "${TASK}" -c "${CONFIG}" "$@"
fi
