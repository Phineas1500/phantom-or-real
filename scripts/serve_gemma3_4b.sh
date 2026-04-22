#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for Gemma 3 4B IT on the local GPU.
#
# Prerequisites:
#   - conda env "phantom" activated
#   - HF_TOKEN env var set and Gemma 3 access accepted at
#     https://huggingface.co/google/gemma-3-4b-it
#
# Starts on http://localhost:8000/v1  (OpenAI-compatible).
# Test with:
#   curl http://localhost:8000/v1/models
#
# Tuned for a single RTX 4090 (24 GB). Gemma 3 4B IT is ~8 GB in bf16; the
# remaining VRAM headroom is used for KV cache / paged attention.

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set. Get one at https://huggingface.co/settings/tokens" >&2
  echo "Then: export HF_TOKEN=hf_..." >&2
  exit 1
fi

MODEL="${MODEL:-google/gemma-3-4b-it}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.85}"

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --dtype bfloat16
