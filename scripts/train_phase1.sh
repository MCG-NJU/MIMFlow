#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/mimflow_l_phase1.yaml}
DATA_ROOT=${DATA_ROOT:-data/imagenet}
OUTPUT_DIR=${OUTPUT_DIR:-workdirs}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
EXPERIMENT=${EXPERIMENT:-mimflow-l-phase1}

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" main.py fit \
  -c "${CONFIG}" \
  --tags.exp "${EXPERIMENT}" \
  --trainer.default_root_dir "${OUTPUT_DIR}" \
  --data.data_root "${DATA_ROOT}" \
  "$@"
