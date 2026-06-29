#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/mimflow_l_phase2_decoder_ft.yaml}
DATA_ROOT=${DATA_ROOT:-data/imagenet}
OUTPUT_DIR=${OUTPUT_DIR:-workdirs}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
EXPERIMENT=${EXPERIMENT:-mimflow-l-phase2-decoder-ft}

if [ $# -lt 1 ]; then
  echo "Usage: $0 PHASE1_CKPT [extra LightningCLI args...]" >&2
  exit 1
fi

PHASE1_CKPT=$1
shift

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" main.py fit \
  -c "${CONFIG}" \
  --tags.exp "${EXPERIMENT}" \
  --trainer.default_root_dir "${OUTPUT_DIR}" \
  --data.data_root "${DATA_ROOT}" \
  --model.stage1_ckpt_path "${PHASE1_CKPT}" \
  "$@"
