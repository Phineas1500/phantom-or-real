#!/bin/bash
# Assemble + deterministically tar the item-WW (attention steering) givemeanode build context.
# Usage: scripts/tables_steer_context.sh <outdir> [pins.json] [frame.jsonl.gz] [tag]   -> prints path, size, sha256
set -euo pipefail
OUT=${1:?outdir}
PINS=${2:-docs/ww_pinned.json}
INPUT=${3:-results/loop_screen/wtq_haystack_input.jsonl.gz}
TAG=${4:-wwctx}
CTX=$OUT/$TAG; rm -rf "$CTX"; mkdir -p "$CTX"
cp scripts/tables_steer_job.py "$CTX/ww_job.py"
cp scripts/wikihop_common.py "$CTX/wikihop_common.py"
cp scripts/wikihop_fake_gemma.py "$CTX/fake_gemma.py"
cp "$INPUT" "$CTX/wikihop_port_input.jsonl.gz"
cp "$PINS" "$CTX/ww_pinned.json"
cat > "$CTX/Dockerfile" <<'DOCKER'
FROM vllm/vllm-openai:v0.26.0
RUN pip install -U --no-cache-dir transformers accelerate
COPY ww_job.py wikihop_common.py fake_gemma.py /app/
COPY wikihop_port_input.jsonl.gz ww_pinned.json /app/
DOCKER
( cd "$CTX" && tar --sort=name --mtime='2026-08-19 00:00:00' --owner=0 --group=0 --numeric-owner -c \
    Dockerfile ww_job.py wikihop_common.py fake_gemma.py wikihop_port_input.jsonl.gz ww_pinned.json ) \
  | zstd -19 -q -f -o "$OUT/$TAG.tar.zst"
echo "$OUT/$TAG.tar.zst $(stat -c %s "$OUT/$TAG.tar.zst") $(sha256sum "$OUT/$TAG.tar.zst" | cut -d' ' -f1)"
