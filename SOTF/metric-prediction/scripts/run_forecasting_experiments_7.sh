#!/usr/bin/env bash
set -euo pipefail

EXPERIMENTS=(
  # weight_statistics
  experiments/forecasting/weight_statistics/final_value/mlp/ds7
  experiments/forecasting/weight_statistics/final_value/xgboost/ds7
  experiments/forecasting/weight_statistics/horizon/mlp/ds7
  experiments/forecasting/weight_statistics/horizon/xgboost/ds7

  # latent_averaged
  # experiments/forecasting/latent_averaged/final_value/mlp/ds7
  # experiments/forecasting/latent_averaged/final_value/ridge/ds7
  # experiments/forecasting/latent_averaged/final_value/xgboost/ds7
  # experiments/forecasting/latent_averaged/horizon/mlp/ds7
  # experiments/forecasting/latent_averaged/horizon/ridge/ds7
  # experiments/forecasting/latent_averaged/horizon/xgboost/ds7

  # # latent_tokens
  # experiments/forecasting/latent_tokens/final_value/transformer/ds7
  # experiments/forecasting/latent_tokens/horizon/transformer/ds7

  # # latent_windows
  # experiments/forecasting/latent_windows/final_value/lstm/ds7
  # experiments/forecasting/latent_windows/horizon/lstm/ds7
)

TOTAL=${#EXPERIMENTS[@]}
FAILED=()

for i in "${!EXPERIMENTS[@]}"; do
  exp="${EXPERIMENTS[$i]}"
  echo "========================================"
  echo "[$((i+1))/$TOTAL] Running: $exp"
  echo "========================================"
  tmplog=$(mktemp)
  if PYTHONPATH="$(pwd)/metric-prediction/src${PYTHONPATH:+:$PYTHONPATH}" \
     uv run metric-prediction/runner.py -cn "$exp" 2>&1 | tee "$tmplog"; then
    echo "[$((i+1))/$TOTAL] Done: $exp"
  else
    echo "[$((i+1))/$TOTAL] FAILED: $exp" >&2
    failed_subs=$(grep -E '^\s+Failed ' "$tmplog" | sed 's/^[[:space:]]*Failed \([^:]*\):.*/\1/' || true)
    if [ -n "$failed_subs" ]; then
      echo "  Failed sub-experiments:" >&2
      while IFS= read -r sub; do
        echo "    - $sub" >&2
      done <<< "$failed_subs"
    fi
    FAILED+=("$exp")
  fi
  rm -f "$tmplog"
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
