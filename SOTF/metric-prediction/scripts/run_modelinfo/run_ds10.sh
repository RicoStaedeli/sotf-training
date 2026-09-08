#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

BATCHES=(
  experiments/forecasting/latent_averaged/final_value/ridge/ds10
  experiments/forecasting/latent_averaged/final_value/xgboost/ds10
  experiments/forecasting/latent_averaged/horizon/mlp/ds10
  experiments/forecasting/latent_averaged/horizon/ridge/ds10
  experiments/forecasting/latent_averaged/horizon/xgboost/ds10
  experiments/forecasting/weight_statistics/final_value/mlp/ds10
  experiments/forecasting/weight_statistics/final_value/xgboost/ds10
  experiments/forecasting/weight_statistics/horizon/mlp/ds10
  experiments/forecasting/weight_statistics/horizon/xgboost/ds10
)

TOTAL=${#BATCHES[@]}
FAILED=()

for i in "${!BATCHES[@]}"; do
  batch="${BATCHES[$i]}"
  echo "[1/$TOTAL] $batch"
  if uv run python "$REPO_ROOT/metric-prediction/scripts/log_model_info.py" -cn "$batch"; then
    echo "  done"
  else
    echo "  FAILED: $batch" >&2
    FAILED+=("$batch")
  fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All $TOTAL batches completed successfully."
else
  echo "${#FAILED[@]} batch(es) failed:"
  for batch in "${FAILED[@]}"; do
    echo "  - $batch"
  done
  exit 1
fi
