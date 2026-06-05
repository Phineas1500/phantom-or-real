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
EXPECTED_JOB_ID=${EXPECTED_JOB_ID:-456693}
REPORT="$REPO/docs/optimized_gold_steering_27b_l45_property_decode_sweep.json"
JSONL="$REPO/results/stage2/steering/optimized_gold_l45_property_decode_sweep.jsonl"
STDOUT="$REPO/slurm_logs/stage2_opt_steer27b_${EXPECTED_JOB_ID}.out"
STDERR="$REPO/slurm_logs/stage2_opt_steer27b_${EXPECTED_JOB_ID}.err"

cd "$REPO"

echo "==== optimized steering afterany checkpoint ===="
date

echo

echo "-- queue --"
squeue -u "$USER" || true

echo

echo "-- row count --"
wc -l "$JSONL" || true

echo

echo "-- latest stdout --"
tail -n 80 "$STDOUT" || true

echo

echo "-- latest stderr --"
tail -n 80 "$STDERR" || true

if [ -f "$REPORT" ] && grep -q "$EXPECTED_JOB_ID" "$REPORT"; then
    echo
    echo "-- optimized report summary --"
    python - <<'PY2'
import json
from pathlib import Path
report = json.loads(Path('docs/optimized_gold_steering_27b_l45_property_decode_sweep.json').read_text())
summary = report.get('summary', {})
noise = report.get('matched_noise_summary', {})
print('slurm_job_id=', report.get('slurm_job_id'))
print('elapsed_seconds=', round(float(report.get('elapsed_seconds', 0.0)), 1))
print('positive_control=', report.get('positive_control_gate', {}).get('status'))
print('matched_noise=', noise)
for name, flips in sorted(summary.get('flips_vs_baseline', {}).items()):
    print(name, flips)
PY2
    python -c 'import json; files=["docs/optimized_gold_steering_27b_l45_property_decode_sweep.json"]; [json.load(open(f)) for f in files]; print("json ok", len(files))'
    git diff --check
    echo "-- optimized checkpoint complete --"
else
    echo
    echo "-- optimized report not ready for expected job ${EXPECTED_JOB_ID} --"
fi
