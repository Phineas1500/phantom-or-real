#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for Qwen3.5 27B.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\": false}}"

exec vllm serve "$MODEL" \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --disable-custom-all-reduce \
  --default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS" \
  --generation-config vllm \
  --dtype bfloat16
