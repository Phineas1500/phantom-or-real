#!/bin/bash
# Assemble + deterministically tar the item-W1 givemeanode build context.
# Usage: scripts/wikihop_w1_context.sh <outdir> [pinned.npz]   -> prints path, size, sha256
# The npz (default results/loop_screen/wikihop_w0_pinned.npz; pass the sweep npz for
# the layer-sweep job) is shipped as /app/wikihop_w0_pinned.npz, the name the job reads.
set -euo pipefail
OUT=${1:?outdir}
NPZ=${2:-results/loop_screen/wikihop_w0_pinned.npz}
CTX=$OUT/w1ctx; rm -rf "$CTX"; mkdir -p "$CTX"
cp scripts/wikihop_w1_job.py "$CTX/w1_job.py"
cp scripts/wikihop_common.py scripts/wikihop_fake_gemma.py "$CTX/"
cp results/loop_screen/wikihop_port_input.jsonl.gz docs/wikihop_w0_pinned.json "$CTX/"
cp "$NPZ" "$CTX/wikihop_w0_pinned.npz"
cat > "$CTX/Dockerfile" <<'DOCKER'
FROM vllm/vllm-openai:v0.26.0
RUN pip install -U --no-cache-dir transformers accelerate
COPY w1_job.py wikihop_common.py wikihop_fake_gemma.py /app/
COPY wikihop_port_input.jsonl.gz wikihop_w0_pinned.npz wikihop_w0_pinned.json /app/
DOCKER
( cd "$CTX" && tar --sort=name --mtime='2026-08-19 00:00:00' --owner=0 --group=0 --numeric-owner -c \
    Dockerfile w1_job.py wikihop_common.py wikihop_fake_gemma.py wikihop_port_input.jsonl.gz wikihop_w0_pinned.npz wikihop_w0_pinned.json ) \
  | zstd -19 -q -f -o "$OUT/w1ctx.tar.zst"
echo "$OUT/w1ctx.tar.zst $(stat -c %s "$OUT/w1ctx.tar.zst") $(sha256sum "$OUT/w1ctx.tar.zst" | cut -d' ' -f1)"
