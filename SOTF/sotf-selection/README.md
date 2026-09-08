# Survival Of The Fittest Selection

This package simulates a population-based training style process on a fixed model zoo.

There is no neural-network training inside this workflow. Instead, it uses:

- a prepared checkpoint dataset from the model zoo
- one row per `(model_id, epoch)` checkpoint
- either
  - a saved metric-forecasting model that predicts a future metric value, or
  - an XGBoost regressor refit online at each decision epoch, or
  - a naive baseline that uses the current metric as the future prediction

The simulation then repeatedly removes the worst models, which gives a "survival of the fittest" selection process.

## What It Does

At each decision round:

1. Start from all alive models.
2. Look at one decision epoch, for example epoch `10`.
3. Predict which models will have the best metric at `epoch + horizon`.
4. Rank the alive models by predicted future performance.
5. Keep only the top fraction, for example the best `50%`.
6. Repeat at the next decision epoch, for example epoch `20`.

Example:

- decision start epoch: `10`
- decision interval: `10`
- prediction horizon: `5`
- survival rate: `0.5`

Then the simulation does:

- at epoch `10`, predict performance at epoch `15`, remove the bottom `50%`
- at epoch `20`, predict performance at epoch `25`, remove the bottom `50%`
- continue until there are no more decision epochs or the minimum population size is reached

## Predictors

The simulation supports three predictor modes.

### `predictor=forecast_model`

This loads the fitted forecasting model from `metric-prediction`.

It expects the saved artifact bundle from a forecasting run:

- `model_bundle.pt` for PyTorch-backed models, or `model_bundle.pkl` for sklearn/XGBoost/LightGBM-style models
- optionally `used_features.json` if needed

The simulation rebuilds the same forecast feature space and applies the fitted preprocessor and regressor to score each checkpoint row.

Important constraints:

- the forecasting model must use `sequence.target_mode=horizon_value`
- the simulation horizon must match the saved forecasting model horizon
- the target column should match the saved forecasting model target column

By default, the simulation reuses the `features` and `sequence` config embedded in the saved model bundle:

- `predictor.use_bundle_feature_config=true`
- `predictor.use_bundle_sequence_config=true`

That is the safest mode, because it ensures the simulation constructs the same feature space that the saved forecasting model expects.

### `predictor=xgboost`

This fits a fresh XGBoost regressor inside the simulation at every decision epoch.

The fit is causal:

- only checkpoint rows whose `epoch + horizon` target would already be observable at that round are used for training
- once a model has been eliminated, later checkpoints from that model are no longer available to subsequent refits

Example with `start_epoch=15`, `interval=5`, and `horizon=5`:

- at epoch `15`, XGBoost is fit on the history observable by epoch `15`
- it predicts the target at epoch `20` for the alive models and removes the worst ones
- at epoch `20`, it is fit again using the larger observed history and the new alive set

Relevant config:

- `predictor=xgboost`
- `predictor.params.*` for XGBoost hyperparameters
- `predictor.min_train_rows`
- `predictor.min_train_models`
- `predictor.fallback_strategy=current_target`
- `predictor.include_observed_metrics=true`
- `predictor.observed_metric_columns=[current_target,test_acc,train_acc,test_loss,train_loss]`

Leakage guardrails for `predictor.observed_metric_columns`:

- future-label columns such as `future_target`, `actual_future_target`, `predicted_future_target`, and `target`
  are rejected
- only values available at the decision epoch are allowed as observed metric inputs

### `predictor=naive`

This does not load or fit any forecasting model.

Current strategy:

- `current_target`: predict that the future metric is equal to the current metric

This gives you a simple baseline for comparison against the forecasting model.

## Data Flow

The simulation uses the same prepared dataset format as `metric-prediction`.

Expected dataset properties:

- prepared wide parquet
- contains at least `sample_idx`, `model_id`, `epoch`, and the configured target column
- one checkpoint row per `(model_id, epoch)`

Internally the flow is:

1. Load the prepared dataset.
2. Apply feature selection.
3. Build forecasting-style sequence features using the configured horizon and lag settings.
4. For each decision epoch, score the alive models.
5. Remove the worst-ranked models.
6. Write round-by-round artifacts.

## Hydra Config Structure

The config root is:

- [config.yaml](/Users/ricostadeli/Documents/HSG/thesis/ms-thesis/sotf-selection/src/sotf_selection/config/config.yaml)

Main config groups:

- `data`
- `features`
- `sequence`
- `selection`
- `predictor`
- `logging`

Most important fields:

### `data`

- `data.parquet_path`: prepared dataset parquet
- `data.target_column`: target metric, for example `test_acc`
- `data.metadata_columns`: metadata columns kept in outputs

### `sequence`

- `sequence.target_mode`: must be `horizon_value`
- `sequence.horizon`: how many epochs ahead to predict
- `sequence.max_lag`: number of lagged feature steps
- `sequence.min_history`: minimum history before a checkpoint can be used

### `selection`

- `selection.start_epoch`: first elimination epoch
- `selection.interval`: how often selection happens
- `selection.survival_rate`: fraction of models kept each round
- `selection.min_population`: stop shrinking below this size
- `selection.metric_goal`: `maximize` or `minimize`
- `selection.eliminate_models_missing_decision_epoch`: if `true`, live models with no checkpoint at a decision epoch are treated as worst-ranked and only retained if needed to satisfy the configured post-round population target; if `false`, all missing-checkpoint models survive that round unchanged

### `predictor`

For naive:

- `predictor=naive`
- `predictor.strategy=current_target`

For forecasting model:

- `predictor=forecast_model`
- `predictor.model_bundle_path=/abs/path/to/model_bundle.pt`
- `predictor.used_features_path=/abs/path/to/used_features.json`

For online XGBoost:

- `predictor=xgboost`
- `predictor.params.n_estimators=400`
- `predictor.params.max_depth=6`
- `predictor.min_train_rows=10`
- `predictor.min_train_models=2`
- `predictor.include_observed_metrics=true`
- `predictor.observed_metric_columns=[current_target,test_acc,train_acc,test_loss,train_loss]`

## How Selection Works

For each decision epoch:

- only currently alive models are considered
- only rows at that decision epoch are scored
- models are ranked by predicted future target
- the target post-round population is `ceil(alive_population * survival_rate)`, subject to `min_population`
- if fewer live models can be scored at that epoch, missing-checkpoint models are retained deterministically to hit that target population

The simulation also records the actual future target value from the dataset, so you can compare:

- predicted ranking
- oracle ranking
- which models would have survived with perfect foresight

## Mathematical Formulation

Let the initial model zoo be

$$
\mathcal{M}=\{1,\dots,N\},
$$

where each model $$i\in\mathcal{M}$$ has checkpoints at epochs

$$
\mathcal{E}_i=\{e_{i,1},e_{i,2},\dots,e_{i,T_i}\}.
$$

At epoch \(e\), model \(i\) has observed target metric \(y_{i,e}\), for example accuracy or loss, and a feature vector

$$
\mathbf{x}_{i,e}\in\mathbb{R}^d.
$$

For horizon-based forecasting with horizon \(h\), the target used for selection is

$$
z_{i,e}=y_{i,e+h}.
$$

For final-value forecasting, the target is instead

$$
z_{i,e}=y_{i,e_i^{\max}},
$$

where \(e_i^{\max}\) is the final available epoch for model \(i\).

Given first selection epoch \(e_0\) and selection interval \(\Delta\), the decision epochs are

$$
\mathcal{D}
=
\left\{
e \in \bigcup_{i\in\mathcal{M}}\mathcal{E}_i
:
e \ge e_0
\land
(e-e_0)\bmod \Delta = 0
\right\}.
$$

Ordered increasingly, write these epochs as

$$
d_1 < d_2 < \dots < d_K.
$$

The alive population starts as

$$
\mathcal{A}_0=\mathcal{M}.
$$

At round \(k\), only alive models with a checkpoint at decision epoch \(d_k\) can be scored:

$$
\mathcal{C}_k
=
\left\{
i\in\mathcal{A}_{k-1}: d_k\in\mathcal{E}_i
\right\}.
$$

The remaining alive models are missing a checkpoint at that decision epoch:

$$
\mathcal{G}_k
=
\mathcal{A}_{k-1}\setminus\mathcal{C}_k.
$$

Each candidate receives a predicted future target score

$$
\hat{z}_{i,k}=f_k(\mathbf{x}_{i,d_k}),
\qquad i\in\mathcal{C}_k.
$$

For `predictor=forecast_model`, \(f_k\) is the saved forecasting model:

$$
\hat{z}_{i,k}=f_\theta(\mathbf{x}_{i,d_k}).
$$

For `predictor=naive`, the future target is approximated by the current target:

$$
\hat{z}_{i,k}=y_{i,d_k}.
$$

For `predictor=xgboost`, a new model is fitted at every decision round using only labels that would already be observable. Let \(o_{i,k}\) be the latest observed epoch for model \(i\) before scoring round \(k\). The causal online training set is

$$
\mathcal{T}_k
=
\left\{
(i,e): e+h \le o_{i,k}
\right\}.
$$

The round-specific model is fitted as

$$
f_k
=
\arg\min_{f\in\mathcal{F}_{\mathrm{XGB}}}
\sum_{(i,e)\in\mathcal{T}_k}
\ell\left(f(\mathbf{x}_{i,e}),z_{i,e}\right).
$$

If there are too few training rows or too few distinct training models, online XGBoost falls back to the naive score:

$$
\hat{z}_{i,k}=y_{i,d_k}.
$$

Let

$$
g=
\begin{cases}
+1, & \text{if the metric is maximized},\\
-1, & \text{if the metric is minimized}.
\end{cases}
$$

Candidates are ranked by \(g\hat{z}_{i,k}\). The predicted rank is

$$
r_{i,k}^{\mathrm{pred}}
=
1+
\left|
\left\{
j\in\mathcal{C}_k:
g\hat{z}_{j,k} > g\hat{z}_{i,k}
\right\}
\right|.
$$

Ties are broken deterministically by `model_id`.

With survival rate \(\rho\in(0,1]\) and minimum population size \(m\), the target population after round \(k\) is

$$
n_k
=
\min
\left(
|\mathcal{A}_{k-1}|,
\max
\left(
m,
\left\lceil \rho |\mathcal{A}_{k-1}| \right\rceil
\right)
\right).
$$

Since only models in \(\mathcal{C}_k\) can be scored, the number of scored models kept is

$$
q_k=\min(|\mathcal{C}_k|,n_k).
$$

The selected scored models are the top-ranked candidates:

$$
\mathcal{S}_k
=
\left\{
i\in\mathcal{C}_k:
r_{i,k}^{\mathrm{pred}}\le q_k
\right\}.
$$

If `selection.eliminate_models_missing_decision_epoch=true`, missing-checkpoint models are treated as worst-ranked and are only retained when needed to reach \(n_k\):

$$
\mathcal{R}_k\subseteq\mathcal{G}_k,
\qquad
|\mathcal{R}_k|
=
\max(0,n_k-|\mathcal{S}_k|).
$$

The implementation chooses \(\mathcal{R}_k\) deterministically by sorted `model_id`. If `selection.eliminate_models_missing_decision_epoch=false`, then

$$
\mathcal{R}_k=\mathcal{G}_k.
$$

The alive population after round \(k\) is

$$
\mathcal{A}_k=\mathcal{S}_k\cup\mathcal{R}_k.
$$

The process stops once

$$
|\mathcal{A}_k|\le m
$$

or when no later decision epochs are available.

For evaluation, the simulation also computes an oracle rank using the actual future target:

$$
r_{i,k}^{\mathrm{oracle}}
=
1+
\left|
\left\{
j\in\mathcal{C}_k:
g z_{j,d_k} > g z_{i,d_k}
\right\}
\right|.
$$

The oracle survivor set for that round is

$$
\mathcal{O}_k
=
\left\{
i\in\mathcal{C}_k:
r_{i,k}^{\mathrm{oracle}}\le q_k
\right\}.
$$

In compact form, the SOTF update is

$$
\mathcal{A}_k
=
\operatorname{Top}_{q_k}
\left(
\mathcal{C}_k,
g\hat{z}_{i,k}
\right)
\cup
\mathcal{R}_k,
$$

where

$$
\mathcal{C}_k
=
\{i\in\mathcal{A}_{k-1}:d_k\in\mathcal{E}_i\},
$$

$$
q_k
=
\min
\left(
|\mathcal{C}_k|,
\min
\left(
|\mathcal{A}_{k-1}|,
\max
\left(
m,
\left\lceil \rho |\mathcal{A}_{k-1}| \right\rceil
\right)
\right)
\right),
$$

and

$$
\hat{z}_{i,k}=f_k(\mathbf{x}_{i,d_k}).
$$

## Outputs

Outputs are written under:

- `sotf-selection/outputs/<date>/<experiment_name>/<time>/`

Main artifacts:

- `resolved_config.json`
  - requested config and effective config actually used
- `simulation_summary.json`
  - high-level result summary
- `round_details.parquet`
  - one row per model per decision round with prediction, ranks, and elimination result
- `round_summary.parquet`
  - one summary row per round
- `final_survivors.parquet`
  - final alive models with their true final target
- `used_features.json`
  - exact feature names used by the selected predictor backend
- log file
  - detailed execution trace

## Running It

If the project environment has been reinstalled after the package changes, you can use:

```bash
uv run sotf-select ...
```

If the console script is not available yet, run the module directly from the source tree:

```bash
PYTHONPATH=sotf-selection/src:metric-prediction/src uv run python -m sotf_selection ...
```

### Example: Naive Baseline

```bash
PYTHONPATH=sotf-selection/src:metric-prediction/src uv run python -m sotf_selection \
  experiment_name=naive_sotf_test_acc \
  data.parquet_path=/absolute/path/to/prepared_dataset__prepared_dataset.parquet \
  data.target_column=test_acc \
  predictor=naive \
  predictor.strategy=current_target \
  sequence.target_mode=horizon_value \
  sequence.horizon=5 \
  selection.start_epoch=10 \
  selection.interval=10 \
  selection.survival_rate=0.5
```

### Example: Forecasting Model Selection

```bash
PYTHONPATH=sotf-selection/src:metric-prediction/src uv run python -m sotf_selection \
  experiment_name=forecast_model_sotf_test_acc \
  data.parquet_path=/absolute/path/to/prepared_dataset__prepared_dataset.parquet \
  data.target_column=test_acc \
  predictor=forecast_model \
  predictor.model_bundle_path=/absolute/path/to/model_bundle.pt \
  predictor.used_features_path=/absolute/path/to/used_features.json \
  sequence.target_mode=horizon_value \
  sequence.horizon=5 \
  selection.start_epoch=10 \
  selection.interval=10 \
  selection.survival_rate=0.5
```

If the saved bundle already contains the correct `features` and `sequence` settings, keep the defaults:

- `predictor.use_bundle_feature_config=true`
- `predictor.use_bundle_sequence_config=true`

### Example: Online XGBoost Selection

```bash
PYTHONPATH=sotf-selection/src:metric-prediction/src uv run python -m sotf_selection \
  experiment_name=xgboost_sotf_test_acc \
  data.parquet_path=/absolute/path/to/prepared_dataset__prepared_dataset.parquet \
  data.target_column=test_acc \
  predictor=xgboost \
  predictor.params.n_estimators=400 \
  predictor.params.max_depth=6 \
  predictor.params.learning_rate=0.05 \
  sequence.target_mode=horizon_value \
  sequence.horizon=5 \
  selection.start_epoch=15 \
  selection.interval=5 \
  selection.survival_rate=0.5
```

## Practical Notes

- Use the forecast-model mode when you want the selection policy to follow your fitted metric-forecasting model.
- Use the online-XGBoost mode when you want the forecast model to be refit inside the simulation from the history observed so far.
- Use the naive mode as a baseline to check whether the forecasting model adds value.
- If your target is a loss instead of an accuracy, set `selection.metric_goal=minimize`.
- If a live model does not have a checkpoint at a decision epoch, its behavior is controlled by `selection.eliminate_models_missing_decision_epoch`.

## Evaluation

The evaluation CLI and documentation live under:

- [src/evaluation/README.md](/Users/ricostadeli/Documents/HSG/thesis/ms-thesis/sotf-selection/src/evaluation/README.md:1)

Preferred command:

```bash
uv run sotf-select-evaluate <experiment_root>
```

The evaluation outputs include:

- `sotf-selection/outputs/evaluation/<name>_<evaluation-id>/...`
- `11_best_survivor_gap_comparison.png`
- `11_best_survivor_gap_by_experiment.png`

## Evaluation Metrics: R2 and RMSE

For predictor quality, the SOTF evaluator compares `predicted_future_target` against `actual_future_target` at each decision epoch.

For a set of rows in one epoch:

$$
\bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i
$$

$$
\mathrm{SSE} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

$$
\mathrm{SST} = \sum_{i=1}^{n} (y_i - \bar{y})^2
$$

$$
R^2 = 1 - \frac{\mathrm{SSE}}{\mathrm{SST}}
$$

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

Where:

- $y_i$ is the true future target.
- $\hat{y}_i$ is the predicted future target.
- $n$ is the number of valid prediction/target pairs at that epoch.

Interpretation notes:

- `R2 = 1` is perfect prediction.
- `R2 = 0` matches a baseline that predicts the epoch-wise mean target.
- `R2 < 0` is worse than that baseline.

Why `R2` can get worse in later epochs even with more training rows:

- In later selection rounds, surviving models are more similar, so target variance (`SST`) is often much smaller.
- With small `SST`, even small absolute errors can reduce `R2` substantially.
- RMSE can stay stable while `R2` drops, so interpret both together.

## Tested Behavior

The current tests cover:

- Hydra config composition
- naive elimination logic
- forecast-model bundle loading and selection
- online XGBoost refitting with causal training history

Relevant test files:

- [test_hydra_config.py](/Users/ricostadeli/Documents/HSG/thesis/ms-thesis/sotf-selection/tests/test_hydra_config.py)
- [test_simulation.py](/Users/ricostadeli/Documents/HSG/thesis/ms-thesis/sotf-selection/tests/test_simulation.py)
