#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${1:-$(pwd)}"
OUT_DIR="outputs_revision/real_corruption_encoder_constructor_pilot_20260924"
SOURCE_ROOT="outputs_revision/protocol_12_full"
CORA_SOURCE="outputs_revision/results/cora_full"
LOG_DIR="$OUT_DIR/logs"
STATUS_DIR="$OUT_DIR/status"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python3.10}"

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
  echo "datasets=Airports-USA Cora Amazon-Photo Roman-Empire"
  echo "backbones=gcn sage gat"
  echo "runs=3"
  echo "corruptions=none:0 degree_preserving:0.25 homophily_attack:0.25"
  echo "encoder_matrix=aux_gcn-vgae vgae-vgae vgae-aux_gcn aux_gcn-aux_gcn"
  echo "constructors=delaunay gabriel rng mst"
  echo "ratios=0 0.25 0.55"
  "$PYTHON_BIN" --version 2>&1
} > "$STATUS_DIR/run_info.txt"

pids=()
for shard in 0 1 2; do
  "$PYTHON_BIN" -u scripts/run_real_corruption_encoder_constructor_ablation.py \
    --source-root "$SOURCE_ROOT" \
    --source-override "Cora=$CORA_SOURCE" \
    --output-dir "$OUT_DIR" \
    --datasets Airports-USA Cora Amazon-Photo Roman-Empire \
    --backbones gcn sage gat \
    --constructors delaunay gabriel rng mst \
    --ratios 0 0.25 0.55 \
    --corruptions none:0 degree_preserving:0.25 homophily_attack:0.25 \
    --runs 3 --seed-start 52000 \
    --downstream-epochs 200 --auxiliary-epochs 150 --vgae-epochs 200 \
    --gate-margin-pp 1.0 --device cpu \
    --num-shards 3 --shard-index "$shard" \
    > "$LOG_DIR/worker_${shard}.log" 2>&1 &
  pids+=("$!")
  echo "$!" > "$STATUS_DIR/worker_${shard}.pid"
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then
  echo "failed_timestamp=$(date --iso-8601=seconds)" > "$STATUS_DIR/FAILED"
  exit 1
fi

"$PYTHON_BIN" -u scripts/run_real_corruption_encoder_constructor_ablation.py \
  --source-root "$SOURCE_ROOT" --source-override "Cora=$CORA_SOURCE" \
  --output-dir "$OUT_DIR" --aggregate-only > "$LOG_DIR/aggregate.log" 2>&1
echo "completed_timestamp=$(date --iso-8601=seconds)" > "$STATUS_DIR/COMPLETED"

