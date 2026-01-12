#!/usr/bin/env bash
set -euo pipefail

source /hy-tmp/align-lab/scripts/env.sh

CFG=${1:-/hy-tmp/align-lab/configs/sft_ultrachat_lora.yaml}
RUN_ID=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="/hy-tmp/align-lab/outputs/runs/${RUN_ID}"

mkdir -p "$RUN_DIR"
cp "$CFG" "$RUN_DIR/config.yaml"

echo "[INFO] RUN_DIR=$RUN_DIR"
echo "[INFO] Using config: $CFG"

# 训练日志保存到 run_dir
python /hy-tmp/sanity_sft_ultrachat_qwen_lora.py \
  --config "$RUN_DIR/config.yaml" \
  --out "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/train.log"

echo "[DONE] logs: $RUN_DIR/train.log"
echo "[DONE] artifacts: $RUN_DIR"
