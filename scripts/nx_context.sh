#!/bin/bash
# Assemble + deterministically tar the item-NX givemeanode build context.
# Usage: scripts/nx_context.sh <outdir> [tag]   -> prints path, size, sha256
# Ships the NX job, the shared prompt module, the fake model for dry runs, the
# WF frame (the rows WX ran on) under the name the job reads, and the WX pins.
set -euo pipefail
OUT=${1:?outdir}
TAG=${2:-nxctx}
CTX=$OUT/$TAG; rm -rf "$CTX"; mkdir -p "$CTX"
cp scripts/nx_neuron_job.py "$CTX/nx_job.py"
cp scripts/wikihop_common.py "$CTX/wikihop_common.py"
cp scripts/wikihop_fake_gemma.py "$CTX/fake_gemma.py"
cp results/loop_screen/wikihop_fresh_input.jsonl.gz "$CTX/wikihop_port_input.jsonl.gz"
cp docs/wikihop_wx_pinned.json "$CTX/wikihop_wx_pinned.json"
cat > "$CTX/Dockerfile" <<'DOCKER'
FROM vllm/vllm-openai:v0.26.0
RUN pip install -U --no-cache-dir transformers accelerate
COPY nx_job.py wikihop_common.py fake_gemma.py /app/
COPY wikihop_port_input.jsonl.gz wikihop_wx_pinned.json /app/
DOCKER
( cd "$CTX" && tar --sort=name --mtime='2026-08-19 00:00:00' --owner=0 --group=0 --numeric-owner -c \
    Dockerfile nx_job.py wikihop_common.py fake_gemma.py wikihop_port_input.jsonl.gz wikihop_wx_pinned.json ) \
  | zstd -19 -q -f -o "$OUT/$TAG.tar.zst"
echo "$OUT/$TAG.tar.zst $(stat -c %s "$OUT/$TAG.tar.zst") $(sha256sum "$OUT/$TAG.tar.zst" | cut -d' ' -f1)"
