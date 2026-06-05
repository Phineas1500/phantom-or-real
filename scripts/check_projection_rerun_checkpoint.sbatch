#!/usr/bin/env bash
#SBATCH -A scholar
#SBATCH --partition=scholar-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

set -euo pipefail

REPO=/scratch/scholar/skiron/phantom-or-real
LABEL=${CHECKPOINT_LABEL:-checkpoint}
EXPECTED_ERROR_JOB=${EXPECTED_ERROR_JOB:-456671}
EXPECTED_RAW_JOB=${EXPECTED_RAW_JOB:-456672}

ERROR_JSONL="$REPO/results/stage2/steering/error_l45_layer_45_width_262k_l0_small_top128_property_decode_sweep.jsonl"
RAW_JSONL="$REPO/results/stage2/steering/raw_l45_property_decode_sweep.jsonl"
ERROR_REPORT="$REPO/docs/error_l45_layer_45_width_262k_l0_small_top128_property_decode_sweep.json"
RAW_REPORT="$REPO/docs/raw_steering_27b_l45_property_decode_sweep.json"

cd "$REPO"

echo "==== projection rerun checkpoint: $LABEL ===="
date

echo
echo "-- queue --"
squeue -u "$USER" || true

echo
echo "-- row counts --"
wc -l "$ERROR_JSONL" "$RAW_JSONL" || true

echo
echo "-- latest error stdout --"
tail -n 25 "slurm_logs/stage2_err_steer27b_${EXPECTED_ERROR_JOB}.out" || true

echo
echo "-- latest raw stdout --"
tail -n 25 "slurm_logs/stage2_raw_steer_dec_${EXPECTED_RAW_JOB}.out" || true

if grep -q "$EXPECTED_ERROR_JOB" "$ERROR_REPORT" && grep -q "$EXPECTED_RAW_JOB" "$RAW_REPORT"; then
  echo
  echo "-- both reports ready: regenerating diagnostics/dashboard --"
  module load conda/2024.09
  conda activate "/scratch/scholar/$USER/conda-envs/phantom"
  python scripts/stage2_steering_effectiveness_diagnostics.py
  python scripts/stage2_causal_abstraction_dashboard.py
  python -c 'import json; files=["docs/steering_effectiveness_diagnostics.json","docs/next_paper_causal_abstraction_schema.json"]; [json.load(open(f)) for f in files]; print("json ok", len(files))'
  git diff --check
  echo "-- next-step status complete --"
else
  echo
  echo "-- reports not both ready yet; skipping diagnostics regen --"
fi
