#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../src/config/evaluation"

cd "${REPO_ROOT}"

skipped=0
ran=0

for mode_dir in "${CONFIG_DIR}"/final_value "${CONFIG_DIR}"/horizon_value; do
  [[ -d "${mode_dir}" ]] || continue
  mode="$(basename "${mode_dir}")"

  for config_file in "${mode_dir}"/ds*.yaml; do
    [[ -f "${config_file}" ]] || continue

    config_name="$(basename "${config_file}" .yaml)"
    batches_dir="$(grep -E '^\s*batches_dir:' "${config_file}" | head -1 | sed 's/.*batches_dir:[[:space:]]*//')"

    if [[ -n "${batches_dir}" && ! -d "${REPO_ROOT}/${batches_dir}" ]]; then
      echo "[skip] ${mode}/${config_name} — no results yet: ${batches_dir}"
      skipped=$((skipped + 1))
      continue
    fi

    ran=$((ran + 1))
    echo
    echo "── ${mode}/${config_name} ──"
    uv run sotf-selection/evaluate_config.py -cn "${mode}/${config_name}"
  done
done

echo
echo "Done: ${ran} evaluated, ${skipped} skipped."
