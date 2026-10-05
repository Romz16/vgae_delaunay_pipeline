#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${1:-$(pwd)}"
OUT_DIR="outputs_revision/corruption_recovery_pilot"
LOG_DIR="$OUT_DIR/logs"
STATUS_DIR="$OUT_DIR/status"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python3.10}"

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR" "$STATUS_DIR"
{
  echo "start_timestamp=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "workers=3"
  echo "device=cpu"
  echo "threads_per_worker=$OMP_NUM_THREADS"
  echo "profile=recovery_pilot"
  "$PYTHON_BIN" --version 2>&1
} > "$STATUS_DIR/run_info.txt"

"$PYTHON_BIN" -u scripts/run_synthetic_suitability.py catalog --profile recovery_pilot --output-dir "$OUT_DIR" \
  > "$LOG_DIR/catalog.log" 2>&1

pids=()
for shard in 0 1 2; do
  "$PYTHON_BIN" -u scripts/run_synthetic_suitability.py run --profile recovery_pilot --output-dir "$OUT_DIR" \
    --num-shards 3 --shard-index "$shard" --no-finalize > "$LOG_DIR/worker_${shard}.log" 2>&1 &
  pids+=("$!")
  echo "$!" > "$STATUS_DIR/worker_${shard}.pid"
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then
  echo "failed_timestamp=$(date --iso-8601=seconds)" > "$STATUS_DIR/FAILED"
  exit 1
fi

"$PYTHON_BIN" -u scripts/run_synthetic_suitability.py run --profile recovery_pilot --output-dir "$OUT_DIR" \
  --max-graphs 0 > "$LOG_DIR/aggregate.log" 2>&1
"$PYTHON_BIN" -u scripts/run_synthetic_suitability.py analyze-recovery --profile recovery_pilot --output-dir "$OUT_DIR" \
  > "$LOG_DIR/analyze_recovery.log" 2>&1
echo "completed_timestamp=$(date --iso-8601=seconds)" > "$STATUS_DIR/COMPLETED"
