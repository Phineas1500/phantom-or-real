#!/bin/bash
# Assemble + deterministically tar an item-NX givemeanode build context.
# Usage: scripts/nx_context.sh <outdir> [tag] [nx_neurons.json [nx_tests.json]]   -> prints path, size, sha256
# Stages 1-3 (no neuron list): the NX job, the shared prompt module, the fake
# model, the WF frame (the rows WX ran on) under the name the job reads, the WX
# pins. Stage 4 (neuron list given): the legibility job instead, plus the
# neuron list (and the judge's test texts for the score pass); datasets is
# installed for the fineweb stream.
set -euo pipefail
OUT=${1:?outdir}
TAG=${2:-nxctx}
NEURONS=${3:-}
TESTS=${4:-}
CTX=$OUT/$TAG; rm -rf "$CTX"; mkdir -p "$CTX"
cp scripts/wikihop_common.py "$CTX/wikihop_common.py"
cp scripts/wikihop_fake_gemma.py "$CTX/fake_gemma.py"
cp results/loop_screen/wikihop_fresh_input.jsonl.gz "$CTX/wikihop_port_input.jsonl.gz"
if [ -z "$NEURONS" ]; then
  cp scripts/nx_neuron_job.py "$CTX/nx_job.py"
  cp docs/wikihop_wx_pinned.json "$CTX/wikihop_wx_pinned.json"
  FILES="nx_job.py wikihop_common.py fake_gemma.py wikihop_port_input.jsonl.gz wikihop_wx_pinned.json"
  PIP="transformers accelerate"
else
  cp scripts/nx_legibility_job.py "$CTX/nx_leg_job.py"
  cp "$NEURONS" "$CTX/nx_neurons.json"
  FILES="nx_leg_job.py wikihop_common.py fake_gemma.py wikihop_port_input.jsonl.gz nx_neurons.json"
  if [ -n "$TESTS" ]; then cp "$TESTS" "$CTX/nx_tests.json"; FILES="$FILES nx_tests.json"; fi
  PIP="transformers accelerate datasets"
fi
{ echo "FROM vllm/vllm-openai:v0.26.0"; echo "RUN pip install -U --no-cache-dir $PIP"; echo "COPY $FILES /app/"; } > "$CTX/Dockerfile"
( cd "$CTX" && tar --sort=name --mtime='2026-08-19 00:00:00' --owner=0 --group=0 --numeric-owner -c Dockerfile $FILES ) \
  | zstd -19 -q -f -o "$OUT/$TAG.tar.zst"
echo "$OUT/$TAG.tar.zst $(stat -c %s "$OUT/$TAG.tar.zst") $(sha256sum "$OUT/$TAG.tar.zst" | cut -d' ' -f1)"
