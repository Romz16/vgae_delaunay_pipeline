#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${1:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python3.10}"
ROOT_OUT="outputs_revision/blind_sequential"
KNOWLEDGE="$ROOT_OUT/knowledge.csv"
BASE_OUTCOMES="outputs_revision/corruption_recovery_pilot/graph_level_outcomes.csv"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"

cd "$PROJECT_DIR"
mkdir -p "$ROOT_OUT/logs" "$ROOT_OUT/status"
{
  echo "start_timestamp=$(date --iso-8601=seconds)"
  echo "protocol=frozen_predictions_before_outcomes"
  echo "workers=3"
  echo "success_definition=mean selected rewiring gain greater than 0.5 pp"
  "$PYTHON_BIN" --version
} > "$ROOT_OUT/status/run_info.txt"

run_wave() {
  local profile="$1" wave="$2"
  local out="outputs_revision/${wave}"
  local predictions="$ROOT_OUT/${wave}_predictions_frozen.csv"
  local score="$ROOT_OUT/${wave}_predictions_scored.csv"
  mkdir -p "$out/logs" "$out/status"

  "$PYTHON_BIN" -u scripts/run_synthetic_suitability.py catalog --profile "$profile" --output-dir "$out" \
    > "$out/logs/catalog.log" 2>&1

  "$PYTHON_BIN" -u scripts/run_blind_sequential.py predict \
    --knowledge "$KNOWLEDGE" --base-outcomes "$BASE_OUTCOMES" \
    --catalog "$out/synthetic_graph_metrics.csv" --target-output-dir "$out" \
    --predictions "$predictions" --wave "$wave" \
    > "$out/logs/blind_predict.log" 2>&1

  local pids=()
  for shard in 0 1 2; do
    "$PYTHON_BIN" -u scripts/run_synthetic_suitability.py run --profile "$profile" --output-dir "$out" \
      --num-shards 3 --shard-index "$shard" --no-finalize \
      > "$out/logs/worker_${shard}.log" 2>&1 &
    pids+=("$!")
    echo "$!" > "$out/status/worker_${shard}.pid"
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if (( failed )); then
    echo "failed_timestamp=$(date --iso-8601=seconds)" > "$out/status/FAILED"
    return 1
  fi

  "$PYTHON_BIN" -u scripts/run_synthetic_suitability.py run --profile "$profile" --output-dir "$out" --max-graphs 0 \
    > "$out/logs/aggregate.log" 2>&1
  "$PYTHON_BIN" -u scripts/run_blind_sequential.py reveal \
    --knowledge "$KNOWLEDGE" --base-outcomes "$BASE_OUTCOMES" \
    --predictions "$predictions" --target-output-dir "$out" --score-output "$score" --wave "$wave" \
    > "$out/logs/blind_reveal.log" 2>&1
  echo "completed_timestamp=$(date --iso-8601=seconds)" > "$out/status/COMPLETED"
}

run_wave blind_wave1 blind_wave1_geometric
run_wave blind_wave2 blind_wave2_degree_corrected
run_wave blind_wave3 blind_wave3_hierarchical

"$PYTHON_BIN" -u scripts/run_blind_sequential.py finalize \
  --knowledge "$KNOWLEDGE" --base-outcomes "$BASE_OUTCOMES" \
  --score-files "$ROOT_OUT/blind_wave1_geometric_predictions_scored.csv" \
                "$ROOT_OUT/blind_wave2_degree_corrected_predictions_scored.csv" \
                "$ROOT_OUT/blind_wave3_hierarchical_predictions_scored.csv" \
  --report-output "$ROOT_OUT/BLIND_SEQUENTIAL_REPORT.md" \
  > "$ROOT_OUT/logs/finalize.log" 2>&1

echo "completed_timestamp=$(date --iso-8601=seconds)" > "$ROOT_OUT/status/COMPLETED"
