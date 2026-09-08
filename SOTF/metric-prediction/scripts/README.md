# End-to-End Weight-Statistic Split Dataset Generation

This README explains the full pipeline that starts in `learning-dynamics` and ends
with the final split-specific weight-statistic `.pt` files in
`metric-prediction/cache/weight_statistic_datasets/`.

## Thesis Metric-Prediction Plots

Generate the naive current-value baseline figure used in the thesis with:

```shell
uv run python metric-prediction/scripts/plot_naive_current_value_baseline.py
```

By default, the script reads the saved forecasting validation predictions from
`metric-prediction/outputs/forecasting/weight_statistics`, uses dataset plot
names from `learning-dynamics/configs/model_zoo_informations.json`, and writes
the PDF and CSV to:

```text
sotf-selection/outputs/metric_prediction/naive_current_value_baseline/
```

Generate the compact predictor-family comparison figure used in the thesis with:

```shell
uv run python metric-prediction/scripts/plot_predictor_family_comparison.py
```

By default, the script reads all forecasting `metrics.json` files, selects
exactly one best configuration per dataset, target mode, and predictor family by
validation RMSE, computes the mean RMSE reduction only from these selected
configurations, compares them against the naive baseline CSV, and writes the
overview PDF, per-predictor dataset PDFs, performance-feature ablation PDF, and
CSV files to:

```text
sotf-selection/outputs/metric_prediction/predictor_family_comparison/
```

The full dataset-preparation pipeline below is used to produce:

- a merged checkpoint-level weight-statistics dataset with target metrics
- a prepared wide-form parquet for `metric-prediction`
- a latent `model_id -> split_name` CSV
- final `train`, `val`, and `test` weight-statistic `.pt` files that use the same
  model split as the latent representation dataset

This section is written so an independent reader can reproduce the pipeline step by
step from the repository root:

```text
/Users/ricostadeli/Documents/HSG/thesis/ms-thesis
```

## Pipeline Overview

The pipeline has seven stages:

1. Build the cached long dataframe in `learning-dynamics`.
2. Compute the long-form weight-statistics parquet in `learning-dynamics`.
3. Merge that parquet with the original training metrics in `metric-prediction`.
4. Convert the merged long-form dataset into the prepared wide-form parquet.
5. *(If needed)* Enrich the prepared dataset with zoo performance metrics.
6. Create the latent split CSV from the latent `train`, `val`, and `test` `.pt` files.
7. Materialize the final split-specific weight-statistic `.pt` files.

Step 5 is only needed when the long-form weight-statistics parquet was generated
without going through the merge in Step 3 (for example, when computing weight
statistics directly via `write_metrics_parquet_for_dataset` without the
`merge_ws_and_train_ds.py` join). In that case the prepared parquet will be missing
`train_acc`, `test_loss`, and `train_loss`; the new
[extract_model_zoo_metadata.py](data_preparation/extract_model_zoo_metadata.py)
script adds them in seconds by reading the model zoo's `result.json` files.

The most important files in the pipeline are:

- [learning-dynamics/README.md](../../learning-dynamics/README.md)
- [learning-dynamics/configs/datasets_dgx.yaml](../../learning-dynamics/configs/datasets_dgx.yaml)
- [learning-dynamics/configs/reports/plot_configs_ws.yaml](../../learning-dynamics/configs/reports/plot_configs_ws.yaml)
- [learning-dynamics/src/nnrun/cli.py](../../learning-dynamics/src/nnrun/cli.py)
- [learning-dynamics/Docs/weight_statistics_analysis.md](../../learning-dynamics/Docs/weight_statistics_analysis.md)
- [learning-dynamics/Docs/weight_statistics_reference.md](../../learning-dynamics/Docs/weight_statistics_reference.md)
- [merge_ws_and_train_ds.py](data_preparation/merge_ws_and_train_ds.py)
- [generate_prepared_dataset.py](data_preparation/generate_prepared_dataset.py)
- [create_latent_split_csv.py](latent_representation/create_latent_split_csv.py)
- [create_weight_statistic_split_pt.py](data_preparation/create_weight_statistic_split_pt.py)

## 1. Prerequisites

Before running any step, install the project environment once from the repository
root:

```shell
uv sync --no-install-project
```

On Linux `x86_64`, the shared `pyproject.toml` is configured so `uv` installs
`torch==2.7.1` from PyTorch's official `cu118` index. This avoids pulling a CUDA
12 wheel on clusters whose NVIDIA driver only supports CUDA 11.x runtime
compatibility.

Two important prerequisites:

- The dataset registry in
  [learning-dynamics/configs/datasets_dgx.yaml](../../learning-dynamics/configs/datasets_dgx.yaml)
  must point to valid model-zoo directories on your machine.
- The latent representation split files must already exist under
  `metric-prediction/cache/latent_representation_datasets/...` as three files:
  `*_train.pt`, `*_val.pt`, and `*_test.pt`.

This README covers the weight-statistic side in detail. It assumes the latent
representation `.pt` files have already been generated by the latent encoding
pipeline.

## 2. Build The Long Dataframe In `learning-dynamics`

The first artifact is the cached long dataframe created by
[learning-dynamics/src/nnrun/io/jsonl.py](../../learning-dynamics/src/nnrun/io/jsonl.py).
This step reads each model zoo's `result.json` files and stores a consolidated parquet
with one row per logged training record.

The CLI entrypoint is
[learning-dynamics/nnrun.py](../../learning-dynamics/nnrun.py), which forwards to
[learning-dynamics/src/nnrun/cli.py](../../learning-dynamics/src/nnrun/cli.py).

### What this step produces

For dataset `12_phase_resnet18_tinyimagenet`, the output is:

```text
learning-dynamics/outputs/cache/long__12_phase_resnet18_tinyimagenet.parquet
```

This parquet is later needed by [merge_ws_and_train_ds.py](data_preparation/merge_ws_and_train_ds.py)
to add:

- `train_acc`
- `test_acc`
- `train_loss`
- `test_loss`
- `hyperparameter_key`

### What to check before running

Open [learning-dynamics/configs/datasets_dgx.yaml](../../learning-dynamics/configs/datasets_dgx.yaml)
and verify that the selected dataset key exists and that its `root_dir` is correct.

For `ds12`, the relevant key is:

```yaml
12_phase_resnet18_tinyimagenet:
  root_dir: /ds2/weight_space_learning/model_zoos/phase_transitions-modelzoo/resnet18_tinyimagenet
  dstype: Seed
```

### Run command

Example for `ds12`:

```shell
uv run python learning-dynamics/nnrun.py cache \
  --datasets learning-dynamics/configs/datasets_dgx.yaml \
  --dataset 12_phase_resnet18_tinyimagenet \
  --cache-dir learning-dynamics/outputs/cache \
  --force
```

Generic form:

```shell
uv run python learning-dynamics/nnrun.py cache \
  --datasets learning-dynamics/configs/datasets_dgx.yaml \
  --dataset <dataset_name> \
  --cache-dir learning-dynamics/outputs/cache \
  --force
```

## 3. Compute The Long-Form Weight-Statistics Parquet In `learning-dynamics`

The actual weight-statistics are computed in
[learning-dynamics/src/nnrun/compute/weight_statistics_compute.py](../../learning-dynamics/src/nnrun/compute/weight_statistics_compute.py)
and orchestrated by
[learning-dynamics/src/nnrun/analyses/weight_statistics.py](../../learning-dynamics/src/nnrun/analyses/weight_statistics.py).

The best conceptual references are:

- [learning-dynamics/Docs/weight_statistics_analysis.md](../../learning-dynamics/Docs/weight_statistics_analysis.md)
- [learning-dynamics/Docs/weight_statistics_reference.md](../../learning-dynamics/Docs/weight_statistics_reference.md)

### What this step does

This step reads the cached long dataframe from Step 2 and computes checkpoint-derived
weight statistics for every `(model_id, epoch)` and every configured statistic.

Important details:

- The output is long-form, not yet the wide feature matrix used by `metric-prediction`.
- The output contains one row per weight statistic value.
- This is the artifact that later gets merged with the training metrics parquet.

### Which config controls this step

The command uses
[learning-dynamics/configs/reports/plot_configs_ws.yaml](../../learning-dynamics/configs/reports/plot_configs_ws.yaml).

Before running, open that file and make sure:

- `datasets:` contains the dataset(s) you want to build
- `analyses[0].name` is `weight_analysis`
- `analyses[0].cache_only` is `True`
- `analyses[0].dataset_split` is `full`

For a single dataset run, it is simplest to keep only one dataset in the `datasets:`
list. For `ds12`, that means uncommenting or adding:

```yaml
datasets:
  - 12_phase_resnet18_tinyimagenet
```

### What this step produces

For `ds12`, the main output is:

```text
learning-dynamics/outputs/cache/weight_metrics/complete_ws_tables/12_phase_resnet18_tinyimagenet_full_all_metrics_long.parquet
```

This filename pattern is important because the downstream merge and preparation
scripts expect it.

### Run command

```shell
uv run python learning-dynamics/nnrun.py report \
  --datasets learning-dynamics/configs/datasets_dgx.yaml \
  --config learning-dynamics/configs/reports/plot_configs_ws.yaml
```

## 4. Merge Weight Statistics With The Original Training Metrics

The long-form weight-statistics parquet from Step 3 does not yet contain the full set
of target metrics required by `metric-prediction`.

That enrichment happens in [merge_ws_and_train_ds.py](data_preparation/merge_ws_and_train_ds.py).

### What this step does

The script joins:

- the long-form weight-statistics parquet from Step 3
- the cached long dataframe from Step 2

Join keys:

- `weight_statistics.epoch` <-> `long_dataframe.training_iteration`
- `weight_statistics.model_id` <-> `long_dataframe.model_name_full`

The script adds:

- `train_loss`
- `test_loss`
- `test_acc`
- `train_acc`
- `hyperparameter_key`

### What this step produces

For `ds12`, the output is:

```text
metric-prediction/cache/full_weight_statistic_datasets/12_phase_resnet18_tinyimagenet_full_all_metrics_long.parquet
```

This is the base dataset used by the next preparation step.

### Run command

```shell
uv run python metric-prediction/scripts/data_preparation/merge_ws_and_train_ds.py \
  learning-dynamics/outputs/cache/weight_metrics/complete_ws_tables/12_phase_resnet18_tinyimagenet_full_all_metrics_long.parquet \
  learning-dynamics/outputs/cache/long__12_phase_resnet18_tinyimagenet.parquet \
  metric-prediction/cache/full_weight_statistic_datasets/
```

Generic form:

```shell
uv run python metric-prediction/scripts/data_preparation/merge_ws_and_train_ds.py \
  learning-dynamics/outputs/cache/weight_metrics/complete_ws_tables/<dataset_name>_full_all_metrics_long.parquet \
  learning-dynamics/outputs/cache/long__<dataset_name>.parquet \
  metric-prediction/cache/full_weight_statistic_datasets/
```

## 5. Generate The Prepared Wide-Form Dataset

The long-form merged parquet is still not the format used by `metric-prediction`
models. It must be converted into the prepared wide-form dataset by
[generate_prepared_dataset.py](data_preparation/generate_prepared_dataset.py).

### What this step does

This script:

- reads the merged long-form parquet from Step 4
- builds partitioned intermediate caches under
  `metric-prediction/cache/partitioned_weight_statistic_datasets/`
- pivots the long-form weight statistics into one wide row per `(model_id, epoch)`
- preserves metadata and target columns
- writes the prepared parquet plus feature metadata

This prepared dataset is what `metric-prediction` uses for prediction and forecasting.

### What this step produces

For `ds12`, the key outputs are:

```text
metric-prediction/cache/prepared_datasets/12_phase_resnet18_tinyimagenet_full_all_metrics_long__prepared_dataset.parquet
metric-prediction/cache/prepared_datasets/12_phase_resnet18_tinyimagenet_full_all_metrics_long__prepared_dataset__feature_metadata.parquet
```

### Run command

```shell
uv run python metric-prediction/scripts/data_preparation/generate_prepared_dataset.py \
  metric-prediction/cache/full_weight_statistic_datasets/12_phase_resnet18_tinyimagenet_full_all_metrics_long.parquet
```

Generic form:

```shell
uv run python metric-prediction/scripts/data_preparation/generate_prepared_dataset.py \
  metric-prediction/cache/full_weight_statistic_datasets/<dataset_name>_full_all_metrics_long.parquet
```

## 5a. Enrich The Prepared Dataset With Zoo Performance Metrics (if needed)

Some datasets skip the `merge_ws_and_train_ds.py` step (Step 4 above) and instead
use a weight-statistics parquet that was computed directly. In that case the prepared
dataset will be missing `train_acc`, `test_loss`, and `train_loss`; only `test_acc` is
present because it was stored as the target during PT generation.

You can check quickly:

```shell
python -c "
import pyarrow.parquet as pq
schema = pq.read_schema(
    'metric-prediction/cache/prepared_datasets/<dataset_name>_full_all_metrics_long__prepared_dataset.parquet'
)
for col in ['test_acc', 'train_acc', 'test_loss', 'train_loss']:
    print(col, ':', col in schema.names)
"
```

If any column prints `False`, run the enrichment step.

### What this step does

[extract_model_zoo_metadata.py](data_preparation/extract_model_zoo_metadata.py)
reads the `result.json` files from the model zoo (one per model, JSONL format with one
record per epoch) and builds a `(model_id, epoch, test_acc, train_acc, test_loss,
train_loss)` parquet. It then joins that table onto the existing prepared parquet via
`(model_id, epoch)`, adding only the columns that are currently absent.

The script is fast — it reads text files, not checkpoint weights.

### What this step produces

A metadata parquet (written once, reusable across re-generations):

```text
metric-prediction/cache/zoo_metadata/<zoo_name>_metadata.parquet
```

The prepared dataset parquet is updated in-place with the new columns.

### Run command

For `ds14` (`14_phase_resnet18_tinyimagenet`):

```shell
uv run python metric-prediction/scripts/data_preparation/extract_model_zoo_metadata.py \
  /ds2/weight_space_learning/model_zoos/phase_transitions-modelzoo/resnet18_tinyimagenet \
  --output metric-prediction/cache/zoo_metadata/14_phase_resnet18_tinyimagenet_metadata.parquet \
  --enrich-prepared-dataset \
    metric-prediction/cache/prepared_datasets/14_phase_resnet18_tinyimagenet_width_4-256_full_all_metrics_long__prepared_dataset.parquet
```

Generic form:

```shell
uv run python metric-prediction/scripts/data_preparation/extract_model_zoo_metadata.py \
  <zoo_root_dir> \
  --output metric-prediction/cache/zoo_metadata/<zoo_name>_metadata.parquet \
  --enrich-prepared-dataset \
    metric-prediction/cache/prepared_datasets/<dataset_name>_full_all_metrics_long__prepared_dataset.parquet
```

You can also separate extraction from enrichment — useful when you want to enrich
multiple prepared datasets from the same zoo:

```shell
# Extract once
uv run python metric-prediction/scripts/data_preparation/extract_model_zoo_metadata.py \
  <zoo_root_dir> \
  --output metric-prediction/cache/zoo_metadata/<zoo_name>_metadata.parquet

# Enrich as many prepared datasets as needed using generate_prepared_dataset.py
uv run python metric-prediction/scripts/data_preparation/generate_prepared_dataset.py \
  metric-prediction/cache/full_weight_statistic_datasets/<dataset_name>_full_all_metrics_long.parquet \
  --metadata-parquet metric-prediction/cache/zoo_metadata/<zoo_name>_metadata.parquet
```

The `--metadata-parquet` flag on `generate_prepared_dataset.py` applies the enrichment
automatically at the end of a fresh generation run — no separate call needed.

### After enrichment: regenerate the split `.pt` files

Once the prepared dataset has the new columns, regenerate the weight-statistic split
`.pt` files so the new metadata is embedded in the PT payload:

```shell
uv run python metric-prediction/scripts/data_preparation/create_weight_statistic_split_pt.py \
  metric-prediction/cache/latent_representation_datasets/<latent_dataset_dir>/averaged/latent_representation_splits.csv \
  --prepared-dataset-path \
    metric-prediction/cache/prepared_datasets/<dataset_name>_full_all_metrics_long__prepared_dataset.parquet
```

For `ds14` specifically:

```shell
uv run python metric-prediction/scripts/data_preparation/create_weight_statistic_split_pt.py \
  metric-prediction/cache/latent_representation_datasets/ds_14_resnet18_tinyimagenet_width_4-256_encoded_checkpoints/averaged/latent_representation_splits.csv \
  --prepared-dataset-path \
    metric-prediction/cache/prepared_datasets/14_phase_resnet18_tinyimagenet_width_4-256_full_all_metrics_long__prepared_dataset.parquet
```

## 6. Create The Latent Split CSV

The final weight-statistic split files do not define their own split. Instead, they
reuse the model-level split already present in the latent representation dataset.

That split CSV is created by
[create_latent_split_csv.py](latent_representation/create_latent_split_csv.py).

### Preconditions

Inside the latent representation dataset directory, you must already have exactly one
file for each split:

- `*_latent_representations_train.pt`
- `*_latent_representations_val.pt`
- `*_latent_representations_test.pt`

For `ds12`, those files exist in:

```text
metric-prediction/cache/latent_representation_datasets/ds_12_resnet18_tinyimagenet_ckpts_encoded_checkpoints/averaged/
```

### What this step does

The script reads the `model_ids` from the three latent `.pt` files and writes a CSV
with two columns:

- `model_id`
- `split_name`

This is the bridge between the latent split and the weight-statistic split.

### What this step produces

For `ds12`, the output is:

```text
metric-prediction/cache/latent_representation_datasets/ds_12_resnet18_tinyimagenet_ckpts_encoded_checkpoints/averaged/latent_representation_splits.csv
```

### Run command

```shell
uv run python metric-prediction/scripts/latent_representation/create_latent_split_csv.py \
  metric-prediction/cache/latent_representation_datasets/ds_12_resnet18_tinyimagenet_ckpts_encoded_checkpoints/averaged
```

Generic form:

```shell
uv run python metric-prediction/scripts/latent_representation/create_latent_split_csv.py \
  metric-prediction/cache/latent_representation_datasets/<latent_dataset_dir>
```

## 7. Materialize The Final Weight-Statistic Split `.pt` Files


The final step is handled by
[create_weight_statistic_split_pt.py](data_preparation/create_weight_statistic_split_pt.py).

### What this step does

The script:

- reads the latent `model_id,split_name` CSV from Step 6
- reads the prepared wide-form parquet from Step 5
- assigns each prepared row to `train`, `val`, or `test` via `model_id`
- matches by exact `model_id` first and then by basename as a fallback
- writes one `.pt` file per split

Each saved `.pt` payload includes:

- `features`
- `feature_names`
- `targets`
- `model_ids`
- `epochs`
- `checkpoint_paths`
- `sample_idx`
- `metadata`

These files are now first-class experiment inputs in `metric-prediction`: for
prediction and forecasting configs you can set `data.pt_train_path` and
`data.pt_val_path` directly instead of using `data.parquet_path` plus
`data.split_csv_path`.

### What this step produces

For `ds12`, the outputs are:

```text
metric-prediction/cache/weight_statistic_datasets/12_phase_resnet18_tinyimagenet_full_all_metrics_long/weight_statistics_train.pt
metric-prediction/cache/weight_statistic_datasets/12_phase_resnet18_tinyimagenet_full_all_metrics_long/weight_statistics_val.pt
metric-prediction/cache/weight_statistic_datasets/12_phase_resnet18_tinyimagenet_full_all_metrics_long/weight_statistics_test.pt
```

### Run command

The simplest form lets the script infer the prepared parquet from the `ds_<id>` token
inside the split CSV path:

```shell
uv run python metric-prediction/scripts/data_preparation/create_weight_statistic_split_pt.py \
  metric-prediction/cache/latent_representation_datasets/ds_12_resnet18_tinyimagenet_ckpts_encoded_checkpoints/averaged/latent_representation_splits.csv
```

If you want to be explicit, pass the prepared parquet path too:

```shell
uv run python metric-prediction/scripts/data_preparation/create_weight_statistic_split_pt.py \
  metric-prediction/cache/latent_representation_datasets/ds_12_resnet18_tinyimagenet_ckpts_encoded_checkpoints/averaged/latent_representation_splits.csv \
  --prepared-dataset-path metric-prediction/cache/prepared_datasets/12_phase_resnet18_tinyimagenet_full_all_metrics_long__prepared_dataset.parquet
```

Useful options:

- `--output-dir <path>` to override the default output directory
- `--target-column <name>` to store a different target column in the saved payload
- `--feature-dtype float64` if you do not want the default `float32` feature tensor
- `--strict` to fail on latent/weight-statistic model mismatches instead of dropping
  unmatched rows

## 8. The Complete Command Sequences

### `ds12` (with merge step — all metrics come from the merge)

For dataset `12_phase_resnet18_tinyimagenet`, which goes through `merge_ws_and_train_ds.py`
to add `train_acc`/`test_loss`/`train_loss`, run:

```shell
uv run python learning-dynamics/nnrun.py cache \
  --datasets learning-dynamics/configs/datasets_dgx.yaml \
  --dataset 12_phase_resnet18_tinyimagenet \
  --cache-dir learning-dynamics/outputs/cache \
  --force

uv run python learning-dynamics/nnrun.py report \
  --datasets learning-dynamics/configs/datasets_dgx.yaml \
  --config learning-dynamics/configs/reports/plot_configs_ws.yaml

uv run python metric-prediction/scripts/data_preparation/merge_ws_and_train_ds.py \
  learning-dynamics/outputs/cache/weight_metrics/complete_ws_tables/12_phase_resnet18_tinyimagenet_full_all_metrics_long.parquet \
  learning-dynamics/outputs/cache/long__12_phase_resnet18_tinyimagenet.parquet \
  metric-prediction/cache/full_weight_statistic_datasets/

uv run python metric-prediction/scripts/data_preparation/generate_prepared_dataset.py \
  metric-prediction/cache/full_weight_statistic_datasets/12_phase_resnet18_tinyimagenet_full_all_metrics_long.parquet

uv run python metric-prediction/scripts/latent_representation/create_latent_split_csv.py \
  metric-prediction/cache/latent_representation_datasets/ds_12_resnet18_tinyimagenet_ckpts_encoded_checkpoints/averaged

uv run python metric-prediction/scripts/data_preparation/create_weight_statistic_split_pt.py \
  metric-prediction/cache/latent_representation_datasets/ds_12_resnet18_tinyimagenet_ckpts_encoded_checkpoints/averaged/latent_representation_splits.csv
```

### `ds14` (without merge step — metrics added via zoo metadata extraction)

For dataset `14_phase_resnet18_tinyimagenet_width_4-256`, whose weight-statistics
parquet was generated directly (skipping the merge), the enrichment step is required:

```shell
# Steps 2–3 already done; weight-stats parquet already at:
# metric-prediction/cache/full_weight_statistic_datasets/14_phase_resnet18_tinyimagenet_width_4-256_full_all_metrics_long.parquet

# Step 4 — generate prepared dataset
uv run python metric-prediction/scripts/data_preparation/generate_prepared_dataset.py \
  metric-prediction/cache/full_weight_statistic_datasets/14_phase_resnet18_tinyimagenet_width_4-256_full_all_metrics_long.parquet

# Step 5a — extract zoo metadata and enrich prepared dataset
uv run python metric-prediction/scripts/data_preparation/extract_model_zoo_metadata.py \
  /ds2/weight_space_learning/model_zoos/phase_transitions-modelzoo/resnet18_tinyimagenet \
  --output metric-prediction/cache/zoo_metadata/14_phase_resnet18_tinyimagenet_metadata.parquet \
  --enrich-prepared-dataset \
    metric-prediction/cache/prepared_datasets/14_phase_resnet18_tinyimagenet_width_4-256_full_all_metrics_long__prepared_dataset.parquet

# Step 6 — latent split CSV already exists; skip if present
uv run python metric-prediction/scripts/latent_representation/create_latent_split_csv.py \
  metric-prediction/cache/latent_representation_datasets/ds_14_resnet18_tinyimagenet_width_4-256_encoded_checkpoints/averaged

# Step 7 — materialize weight-statistic split .pt files
uv run python metric-prediction/scripts/data_preparation/create_weight_statistic_split_pt.py \
  metric-prediction/cache/latent_representation_datasets/ds_14_resnet18_tinyimagenet_width_4-256_encoded_checkpoints/averaged/latent_representation_splits.csv \
  --prepared-dataset-path \
    metric-prediction/cache/prepared_datasets/14_phase_resnet18_tinyimagenet_width_4-256_full_all_metrics_long__prepared_dataset.parquet
```

## 9. How To Verify That The Pipeline Finished Correctly

At the end, you should have the following files:

```text
learning-dynamics/outputs/cache/long__12_phase_resnet18_tinyimagenet.parquet
learning-dynamics/outputs/cache/weight_metrics/complete_ws_tables/12_phase_resnet18_tinyimagenet_full_all_metrics_long.parquet
metric-prediction/cache/full_weight_statistic_datasets/12_phase_resnet18_tinyimagenet_full_all_metrics_long.parquet
metric-prediction/cache/prepared_datasets/12_phase_resnet18_tinyimagenet_full_all_metrics_long__prepared_dataset.parquet
metric-prediction/cache/latent_representation_datasets/ds_12_resnet18_tinyimagenet_ckpts_encoded_checkpoints/averaged/latent_representation_splits.csv
metric-prediction/cache/weight_statistic_datasets/12_phase_resnet18_tinyimagenet_full_all_metrics_long/weight_statistics_train.pt
metric-prediction/cache/weight_statistic_datasets/12_phase_resnet18_tinyimagenet_full_all_metrics_long/weight_statistics_val.pt
metric-prediction/cache/weight_statistic_datasets/12_phase_resnet18_tinyimagenet_full_all_metrics_long/weight_statistics_test.pt
```

You can quickly inspect the final directory with:

```shell
find metric-prediction/cache/weight_statistic_datasets/12_phase_resnet18_tinyimagenet_full_all_metrics_long -maxdepth 1 -type f | sort
```

For datasets that went through Step 5a, also verify that the performance metric columns
are present in the prepared parquet and embedded in the PT files:

```shell
# Check prepared parquet columns
python -c "
import pyarrow.parquet as pq
schema = pq.read_schema(
    'metric-prediction/cache/prepared_datasets/'
    '<dataset_name>_full_all_metrics_long__prepared_dataset.parquet'
)
for col in ['test_acc', 'train_acc', 'test_loss', 'train_loss']:
    print(col, ':', col in schema.names)
"

# Check PT metadata keys
python -c "
import torch
pt = torch.load(
    'metric-prediction/cache/weight_statistic_datasets/<dataset_name>/weight_statistics_train.pt',
    map_location='cpu', weights_only=False,
)
print('metadata keys:', list(pt.get('metadata', {}).keys()))
"
```

All four metric columns should print `True` / appear in the metadata keys.

## 10. Common Failure Modes

### Dataset registry points to the wrong machine path

If Step 2 fails early, first verify the absolute `root_dir` values in
[learning-dynamics/configs/datasets_dgx.yaml](../../learning-dynamics/configs/datasets_dgx.yaml).

### `plot_configs_ws.yaml` still points to the wrong dataset

Step 3 does not take `--dataset` directly. It uses the dataset list inside
[plot_configs_ws.yaml](../../learning-dynamics/configs/reports/plot_configs_ws.yaml).
If you forget to update that file, `nnrun.py report` may compute the wrong dataset or
skip the one you wanted.

### Latent directory does not contain exactly one split file per split

[create_latent_split_csv.py](latent_representation/create_latent_split_csv.py) expects
exactly one `train`, one `val`, and one `test` latent `.pt` file in the target
directory.

### Split CSV and prepared dataset disagree on model membership

[create_weight_statistic_split_pt.py](data_preparation/create_weight_statistic_split_pt.py)
will drop unmatched weight-statistic rows by default and log a warning. Use
`--strict` if you want the pipeline to fail instead.

### Forecasting experiments warn `skipping ['train_acc', 'test_loss', 'train_loss'] — not found in dataset metadata`

The `_with_perf` forecasting experiments request observed lag features for
`train_acc`, `test_loss`, and `train_loss`. This warning means those columns are
absent from the PT file metadata.

Cause: the prepared dataset was built without going through either the merge step
(Step 4) or the zoo-metadata enrichment step (Step 5a), so the PT payload only
carries `test_acc`.

Fix: run Step 5a ([extract_model_zoo_metadata.py](data_preparation/extract_model_zoo_metadata.py))
to enrich the prepared dataset, then re-run Step 7 to regenerate the PT files with
the new metadata embedded.
