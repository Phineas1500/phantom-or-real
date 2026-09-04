#!/bin/bash
# Assemble + deterministically tar the item-WK stage-2 givemeanode build context.
# Usage: scripts/wikihop_wk_context.sh <outdir> [pins.json] [stage2_input.jsonl.gz] [tag]   -> prints path, size, sha256
# Ships the WH job (frozen-write + loop mode), the shared prompt module, the fake
# model for dry runs, the stage-2 input (NQ-Swap rows + WikiHop donor rows) under
# the name the job reads, the WK pins, and the real-text W0 gauge npz.
set -euo pipefail
OUT=${1:?outdir}
PINS=${2:-docs/wikihop_wk_pinned.json}
INPUT=${3:-results/loop_screen/wk_stage2_input.jsonl.gz}
TAG=${4:-wkctx2}
NPZ=${5:-results/loop_screen/wikihop_w0_pinned.npz}
CTX=$OUT/$TAG; rm -rf "$CTX"; mkdir -p "$CTX"
cp scripts/wikihop_wh_job.py "$CTX/wh_job.py"
cp scripts/wikihop_common.py "$CTX/wikihop_common.py"
cp scripts/wikihop_fake_gemma.py "$CTX/fake_gemma.py"
cp "$INPUT" "$CTX/wikihop_port_input.jsonl.gz"
cp "$PINS" "$CTX/wikihop_wk_pinned.json"
cp "$NPZ" "$CTX/wikihop_w0_pinned.npz"
cat > "$CTX/Dockerfile" <<'DOCKER'
FROM vllm/vllm-openai:v0.26.0
RUN pip install -U --no-cache-dir transformers accelerate
COPY wh_job.py wikihop_common.py fake_gemma.py /app/
COPY wikihop_port_input.jsonl.gz wikihop_wk_pinned.json wikihop_w0_pinned.npz /app/
DOCKER
( cd "$CTX" && tar --sort=name --mtime='2026-08-19 00:00:00' --owner=0 --group=0 --numeric-owner -c \
    Dockerfile wh_job.py wikihop_common.py fake_gemma.py wikihop_port_input.jsonl.gz wikihop_wk_pinned.json wikihop_w0_pinned.npz ) \
  | zstd -19 -q -f -o "$OUT/$TAG.tar.zst"
echo "$OUT/$TAG.tar.zst $(stat -c %s "$OUT/$TAG.tar.zst") $(sha256sum "$OUT/$TAG.tar.zst" | cut -d' ' -f1)"
