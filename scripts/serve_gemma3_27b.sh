#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for Gemma 3 27B IT.

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set. Gemma access requires an accepted HF token." >&2
  exit 1
fi

MODEL="${MODEL:-google/gemma-3-27b-it}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"

exec vllm serve "$MODEL"   --served-model-name "$MODEL"   --host 0.0.0.0   --port "$PORT"   --max-model-len "$MAX_MODEL_LEN"   --gpu-memory-utilization "$GPU_UTIL"   --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"   --disable-custom-all-reduce   --generation-config vllm   --dtype bfloat16
