#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

BATCHES=(
  experiments/forecasting/latent_averaged/final_value/ridge/ds8
  experiments/forecasting/latent_averaged/final_value/xgboost/ds8
  experiments/forecasting/latent_averaged/horizon/mlp/ds8
  experiments/forecasting/latent_averaged/horizon/ridge/ds8
  experiments/forecasting/latent_averaged/horizon/xgboost/ds8
  experiments/forecasting/latent_windows/final_value/lstm/ds8
  experiments/forecasting/latent_windows/horizon/lstm/ds8
  experiments/forecasting/weight_statistics/final_value/mlp/ds8
  experiments/forecasting/weight_statistics/final_value/xgboost/ds8
  experiments/forecasting/weight_statistics/horizon/mlp/ds8
  experiments/forecasting/weight_statistics/horizon/xgboost/ds8
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
