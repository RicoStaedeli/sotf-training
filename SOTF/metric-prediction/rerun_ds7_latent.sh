#!/usr/bin/env bash
# Re-run all ViT (ds7) latent forecasting experiments after epoch patch.
# Run from the repo root: bash metric-prediction/rerun_ds7_latent.sh

set -euo pipefail

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

BATCHES=(
  "metric-prediction/outputs/forecasting/latent_averaged/final_value/mlp/ds7/batch_files/ds7.yaml"
  "metric-prediction/outputs/forecasting/latent_averaged/final_value/ridge/ds7/batch_files/ds7.yaml"
  "metric-prediction/outputs/forecasting/latent_averaged/final_value/xgboost/ds7/batch_files/ds7.yaml"
  "metric-prediction/outputs/forecasting/latent_averaged/horizon/mlp/ds7/batch_files/ds7.yaml"
  "metric-prediction/outputs/forecasting/latent_averaged/horizon/ridge/ds7/batch_files/ds7.yaml"
  "metric-prediction/outputs/forecasting/latent_averaged/horizon/xgboost/ds7/batch_files/ds7.yaml"
  "metric-prediction/outputs/forecasting/latent_windows/final_value/lstm/ds7/batch_files/ds7.yaml"
  "metric-prediction/outputs/forecasting/latent_windows/horizon/lstm/ds7/batch_files/ds7.yaml"
  "metric-prediction/outputs/forecasting/latent_tokens/final_value/transformer/ds7/batch_files/ds7.yaml"
  "metric-prediction/outputs/forecasting/latent_tokens/horizon/transformer/ds7/batch_files/ds7.yaml"
)

TOTAL=${#BATCHES[@]}
i=1
for src in "${BATCHES[@]}"; do
  name=$(echo "$src" \
    | sed 's|metric-prediction/outputs/forecasting/||' \
    | sed 's|/ds7/batch_files/ds7.yaml||' \
    | tr '/' '_')
  tmp="$TMPDIR/${name}.yaml"
  cp "$src" "$tmp"

  echo ""
  echo "[$i/$TOTAL] $name"
  uv run python metric-prediction/batch.py "$tmp"
  i=$((i + 1))
done

echo ""
echo "All $TOTAL batches done."
