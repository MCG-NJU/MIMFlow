#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/mimflow_l_validate_samples.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-workdirs}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
EXPERIMENT=${EXPERIMENT:-mimflow-l-samples}

if [ $# -lt 1 ]; then
  echo "Usage: $0 CKPT_PATH [extra LightningCLI args...]" >&2
  exit 1
fi

CKPT_PATH=$1
shift

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" main.py validate \
  -c "${CONFIG}" \
  --tags.exp "${EXPERIMENT}" \
  --trainer.default_root_dir "${OUTPUT_DIR}" \
  --ckpt_path "${CKPT_PATH}" \
  "$@"
