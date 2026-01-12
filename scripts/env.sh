#!/usr/bin/env bash
set -e

# cache 全放 /hy-tmp，避免写到 overlay
export HF_HOME=/hy-tmp/align-lab/cache/huggingface
export TRANSFORMERS_CACHE=/hy-tmp/align-lab/cache/huggingface
export TORCH_HOME=/hy-tmp/align-lab/cache/torch

# CUDA allocator：避免碎片化 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

mkdir -p "$HF_HOME" "$TORCH_HOME"
