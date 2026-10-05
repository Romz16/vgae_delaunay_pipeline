#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${1:-$(pwd)}"
OUT_DIR="outputs_revision/synthetic_suitability_pilot"
LOG_DIR="$OUT_DIR/logs"
STATUS_DIR="$OUT_DIR/status"

# The shared RTX 3090 may be occupied by other laboratory users. These small
# graphs are CPU-friendly, so three bounded workers use the 48-core host
# without contending for GPU memory.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR" "$STATUS_DIR"

{
  echo "start_timestamp=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "container=romulo-gnn"
  echo "project_directory=$PROJECT_DIR"
  echo "workers=3"
  echo "device=cpu"
  echo "threads_per_worker=$OMP_NUM_THREADS"
  echo "profile=pilot"
  echo "command=scripts/run_synthetic_pilot_3workers.sh $PROJECT_DIR"
  python --version 2>&1
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
} > "$STATUS_DIR/run_info.txt"

python -u scripts/run_synthetic_suitability.py catalog \
  --profile pilot --output-dir "$OUT_DIR" \
  > "$LOG_DIR/catalog.log" 2>&1

pids=()
for shard in 0 1 2; do
  python -u scripts/run_synthetic_suitability.py run \
    --profile pilot --output-dir "$OUT_DIR" \
    --num-shards 3 --shard-index "$shard" --no-finalize \
    > "$LOG_DIR/worker_${shard}.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "$pid" > "$STATUS_DIR/worker_${shard}.pid"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if (( failed )); then
  echo "failed_timestamp=$(date --iso-8601=seconds)" > "$STATUS_DIR/FAILED"
  exit 1
fi

python -u scripts/run_synthetic_suitability.py run \
  --profile pilot --output-dir "$OUT_DIR" \
  --max-graphs 0 \
  > "$LOG_DIR/aggregate.log" 2>&1
python -u scripts/run_synthetic_suitability.py fit \
  --profile pilot --output-dir "$OUT_DIR" \
  > "$LOG_DIR/fit.log" 2>&1
python -u scripts/run_synthetic_suitability.py validate-real \
  --profile pilot --output-dir "$OUT_DIR" \
  > "$LOG_DIR/validate_real.log" 2>&1
python -u scripts/run_synthetic_suitability.py report \
  --profile pilot --output-dir "$OUT_DIR" \
  > "$LOG_DIR/report.log" 2>&1

echo "completed_timestamp=$(date --iso-8601=seconds)" > "$STATUS_DIR/COMPLETED"
