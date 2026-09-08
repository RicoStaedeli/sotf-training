#!/usr/bin/env bash
set -euo pipefail

EXPERIMENTS=(
  # weight_statistics
  # experiments/forecasting/weight_statistics/final_value/mlp/ds14
  # experiments/forecasting/weight_statistics/final_value/xgboost/ds14
  # experiments/forecasting/weight_statistics/horizon/mlp/ds14
  # experiments/forecasting/weight_statistics/horizon/xgboost/ds14

  # latent_averaged
  # experiments/forecasting/latent_averaged/final_value/mlp/ds14
  # experiments/forecasting/latent_averaged/final_value/ridge/ds14
  # experiments/forecasting/latent_averaged/final_value/xgboost/ds14
  # experiments/forecasting/latent_averaged/horizon/mlp/ds14
  # experiments/forecasting/latent_averaged/horizon/ridge/ds14
  # experiments/forecasting/latent_averaged/horizon/xgboost/ds14

  # latent_tokens
  experiments/forecasting/latent_tokens/final_value/transformer/ds14
  experiments/forecasting/latent_tokens/horizon/transformer/ds14

  # latent_windows
  # experiments/forecasting/latent_windows/final_value/lstm/ds14
  # experiments/forecasting/latent_windows/horizon/lstm/ds14
)

TOTAL=${#EXPERIMENTS[@]}
FAILED=()

for i in "${!EXPERIMENTS[@]}"; do
  exp="${EXPERIMENTS[$i]}"
  echo "========================================"
  echo "[$((i+1))/$TOTAL] Running: $exp"
  echo "========================================"
  if PYTHONPATH="$(pwd)/metric-prediction/src${PYTHONPATH:+:$PYTHONPATH}" \
     uv run metric-prediction/runner.py -cn "$exp"; then
    echo "[$((i+1))/$TOTAL] Done: $exp"
  else
    echo "[$((i+1))/$TOTAL] FAILED: $exp" >&2
    FAILED+=("$exp")
  fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All $TOTAL experiments completed successfully."
else
  echo "${#FAILED[@]} experiment(s) failed:"
  for exp in "${FAILED[@]}"; do
    echo "  - $exp"
  done
  exit 1
fi
