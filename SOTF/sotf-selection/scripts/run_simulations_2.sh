#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

EXPERIMENTS=(
  # ds2 — latent_averaged
  experiments/forecasting/latent_averaged/mlp/ds2
  experiments/forecasting/latent_averaged/ridge/ds2
  experiments/forecasting/latent_averaged/xgboost/ds2
  # ds2 — latent_tokens
  experiments/forecasting/latent_tokens/final_value/lstm/ds2
  experiments/forecasting/latent_tokens/final_value/transformer/ds2
  experiments/forecasting/latent_tokens/horizon/lstm/ds2
  experiments/forecasting/latent_tokens/horizon/transformer/ds2
  # ds2 — latent_windows
  experiments/forecasting/latent_windows/final_value/lstm/ds2
  experiments/forecasting/latent_windows/horizon/lstm/ds2
  # ds2 — weight_statistics
  experiments/forecasting/weight_statistics/final_value/xgboost/ds2
  experiments/forecasting/weight_statistics/horizon/xgboost/ds2
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
