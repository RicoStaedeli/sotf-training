#!/usr/bin/env bash
set -euo pipefail

EXPERIMENTS=(
  experiments/forecasting/exp_9_latent_tokens_lstm_final_ds9
  experiments/forecasting/exp_9_latent_tokens_lstm_final_ds1
  experiments/forecasting/exp_9_latent_tokens_lstm_final_ds2
  experiments/forecasting/exp_9_latent_tokens_lstm_final_ds3
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
