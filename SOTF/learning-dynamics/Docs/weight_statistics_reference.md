# Weight Statistics Reference

---

This document explains every weight statistic currently computed in `learning-dynamics/src/nnrun/compute/weight_statistics_compute.py`.

It is meant as a practical reference:

- what each metric is
- how it is calculated
- a small example
- why it was chosen
- what can be learned from it

---

## Overview

The pipeline computes statistics from saved model checkpoints. It works at two levels:

- **Layer level:** a statistic is computed for one parameter group, usually a layer weight tensor and optionally its bias.
- **Model level:** layer-level statistics are aggregated into one value per model checkpoint.

The main metric families are:

- `norm`
- `distribution`
- `matrix`
- `sharpness`
- `distance`

There are also different scopes:

- `weight_only`: only the weight tensor is used
- `bias_only`: only the bias tensor is used
- `combined`: weight and bias are combined
- `full_model`: the full flattened checkpoint vector is used
- `default`: used for metrics that do not have a weight/bias scope distinction in the current implementation

---

## Relation To Unterthiner et al. (2021)

`weight_statistics_compute.py` now includes the checkpoint statistics used in Unterthiner et al., *Predicting Neural Network Accuracy from Weights*, as far as they belong in a weight-statistics pipeline.

The paper uses four feature blocks:

- architecture encoding
- simple statistics over the full flattened parameter vector
- the same simple statistics per layer, with kernels and biases treated separately
- per-layer `l1` and `l2` norms, again with kernels and biases treated separately

This file now covers the weight-derived parts of that list:

- full-model distribution statistics via scope `full_model`
- per-layer distribution statistics via scopes `weight_only` and `bias_only`
- per-layer norms for weights and biases via scopes `weight_only` and `bias_only`

It does not implement architecture encoding here, because that is metadata rather than a weight statistic.

---

## How The Pipeline Treats Parameters

Checkpoint parameters are grouped by name:

- `layer.weight` and `layer.bias` belong to the same parameter group
- standalone tensors such as embeddings, tokens, or normalization parameters can also become their own groups

For many metrics, tensors are flattened into vectors. For matrix-based metrics, tensors are reshaped:

- 2D tensors stay 2D
- tensors with 3 or more dimensions are reshaped to `(out_dim, -1)`
- 1D tensors become `(1, n)`
- scalars become `(1, 1)`

This means the matrix statistics are designed to be architecture-agnostic enough to work across CNNs, MLPs, ViTs, embeddings, and normalization layers.

---

## Model-Level Aggregation

Some statistics are first computed per layer and then aggregated to the full model.

### Norm Aggregation

If a model has per-layer norms `l1_i`, `l2_i`, `linf_i`, then:

- model `l1` = sum of per-layer `l1`
- model `l2` = `sqrt(sum_i l2_i^2)`
- model `linf` = max of per-layer `linf`

This is mathematically consistent with concatenating all parameters into one long vector.

### Matrix Aggregation

For each per-layer matrix metric `foo`, the model gets:

- `foo_mean`: average across layers
- `foo_max`: maximum across layers

Examples:

- `fro_norm_mean`
- `max_abs_max`
- `row_l2_mean_mean`

### Sharpness Aggregation

For each per-layer sharpness proxy `bar`, the model gets:

- `bar_mean`: average across layers
- `bar_max`: maximum across layers

---

## Metric Family: `norm`

These metrics measure the overall size of parameters.

They are simple, cheap, and often surprisingly informative because many training effects show up as changes in scale.

### `l1`

**Definition**

$$
\|x\|_1 = \sum_j |x_j|
$$

**Example**

- for `x = [1, -2, 3]`
- `l1 = |1| + |−2| + |3| = 6`

**Why chosen**

- captures total absolute magnitude
- more robust to single large coordinates than `linf`
- easy to compare across checkpoints

**What can be learned**

- whether a model tends to spread weight mass across many parameters
- whether training systematically increases or decreases overall parameter magnitude
- whether better models are associated with larger or smaller absolute parameter budgets

**In the context of Unterthiner et al.**

- `l1` is one of the paper's explicit per-layer norm features
- separating weights and biases makes it easier to see whether sparsity-like effects happen in kernels, biases, or both

### `l2`

**Definition**

$$
\|x\|_2 = \sqrt{\sum_j x_j^2}
$$

**Example**

- for `x = [3, 4]`
- `l2 = sqrt(3^2 + 4^2) = 5`

**Why chosen**

- standard Euclidean scale measure
- directly related to many optimization and regularization ideas
- interpretable for both layers and whole-model parameter vectors

**What can be learned**

- how far a model has grown in magnitude
- whether training concentrates energy globally
- whether performance/rank changes track simple parameter-scale changes

**In the context of Unterthiner et al.**

- `l2` is one of the paper's explicit per-layer summary features
- the paper uses it separately for kernels and biases because they can behave differently during training

### `linf`

**Definition**

$$
\|x\|_\infty = \max_j |x_j|
$$

**Example**

- for `x = [1, -7, 2]`
- `linf = 7`

**Why chosen**

- detects whether a single coordinate becomes dominant
- complements `l1` and `l2`

**What can be learned**

- whether the model develops extreme parameter values
- whether instability or specialization is driven by a few parameters instead of broad scale changes

---

## Metric Family: `distribution`

These metrics describe the value distribution of a flattened tensor directly.

This family is the closest match to the "simple statistics" feature block in Unterthiner et al. The paper uses summary statistics of the raw parameter values because they are cheap, architecture-agnostic, and often surprisingly predictive of downstream model quality.

The same family is computed at:

- per-layer `weight_only`
- per-layer `bias_only`
- full-model `full_model`

Use the example vector:

```text
x = [-2, 0, 1, 5]
```

### `mean`

**Definition**

- arithmetic average

$$
{mean}(x) = \frac{1}{n}\sum_j x_j
$$

**Example**

- `mean = (-2 + 0 + 1 + 5) / 4 = 1`

**Why chosen**

- detects whether parameters are centered or shifted away from zero
- can reveal asymmetric drift in the weight distribution

**What can be learned**

- whether optimization creates a positive or negative bias in parameter values
- whether a layer remains centered around zero or develops a systematic offset

### `variance`

**Definition**

- population variance of flattened values

$$
{Var}(x) = \frac{1}{n}\sum_j \left(x_j - {mean}(x)\right)^2
$$

**Example**

- for `x = [-2, 0, 1, 5]`, the mean is `1`
- squared deviations are `[9, 1, 0, 16]`
- `variance = 26 / 4 = 6.5`

**Why chosen**

- measures spread in a scale-sensitive but size-normalized way
- one of the core simple statistics used in Unterthiner et al.

**What can be learned**

- whether a layer is becoming more dispersed over training
- whether two models with similar norms differ in how concentrated or diffuse their weights are

### `percentile_0`

**Definition**

- minimum value of the flattened tensor

**Example**

- for `x = [-2, 0, 1, 5]`, `percentile_0 = -2`

**Why chosen**

- exposes the lower tail of the distribution
- useful when negative outliers matter

**What can be learned**

- whether the parameter distribution develops more extreme negative values
- whether training changes the tails, not just the center

### `percentile_25`

**Definition**

- first quartile of the flattened values

**Example**

- for `x = [-2, 0, 1, 5]`, `percentile_25` lies between the lower values and is about `-0.5` under linear interpolation

**Why chosen**

- describes the lower-middle part of the distribution
- more robust than the minimum

**What can be learned**

- whether the lower half of the distribution shifts over time
- whether changes are broad or only due to a few outliers

### `percentile_50`

**Definition**

- median of the flattened values

**Example**

- for `x = [-2, 0, 1, 5]`, `percentile_50 = 0.5`

**Why chosen**

- robust location statistic
- less sensitive to outliers than the mean

**What can be learned**

- whether the typical parameter value moves even when extremes are noisy
- whether the distribution is skewed when compared to the mean

### `percentile_75`

**Definition**

- third quartile of the flattened values

**Example**

- for `x = [-2, 0, 1, 5]`, `percentile_75` is about `2`

**Why chosen**

- mirrors `percentile_25` on the upper side
- helps describe shape beyond center and variance

**What can be learned**

- whether the upper half of the distribution expands or shifts
- whether stronger models develop systematically larger positive coordinates

### `percentile_100`

**Definition**

- maximum value of the flattened tensor

**Example**

- for `x = [-2, 0, 1, 5]`, `percentile_100 = 5`

**Why chosen**

- exposes the upper tail directly
- useful for spotting extreme positive values

**What can be learned**

- whether training creates large positive outliers
- whether the parameter distribution becomes more skewed or heavy-tailed

### `range_width`

**Definition**

- difference between maximum and minimum

$$
{range\_width}(x) = {percentile}_{100}(x) - {percentile}_{0}(x)
$$

**Example**

- for `x = [-2, 0, 1, 5]`, `range_width = 5 - (-2) = 7`

**Why chosen**

- compact summary of total spread
- derived directly from the percentile endpoints already used above

**What can be learned**

- whether the distribution is widening overall
- whether tail growth is symmetric or driven mostly by one side when interpreted together with the percentiles

---

## Metric Family: `matrix`

These metrics treat a parameter tensor as a matrix `W`.

They are intended to capture size, shape, concentration, and row/column structure.

Use the example matrix:

```text
W = [[ 1, -2],
     [ 2,  0]]
```

### `fro_norm`

**Definition**

- Frobenius norm

$$
\|W\|_F = \sqrt{\sum_{i,j} W_{ij}^2}
$$

**Example**

- `sqrt(1^2 + (-2)^2 + 2^2 + 0^2) = sqrt(9) = 3`

**Why chosen**

- matrix analogue of `l2`
- very standard summary of matrix energy

**What can be learned**

- whether a layer’s total energy grows or shrinks
- whether some architectures accumulate more weight energy in certain layers

### `mean_abs`

**Definition**

- average absolute entry magnitude

$$
{mean\_abs}(W) = {mean}(|W_{ij}|)
$$

**Example**

- absolute values are `[1, 2, 2, 0]`
- `mean_abs = 1.25`

**Why chosen**

- separates scale from layer size better than raw norm
- easy to compare across differently sized layers

**What can be learned**

- whether parameters become uniformly larger or smaller on average
- whether a layer is dense with moderate weights rather than driven by a few large ones

### `max_abs`

**Definition**

$$
{max\_abs}(W) = \max_{i,j} |W_{ij}|
$$

**Example**

- `max_abs = 2`

**Why chosen**

- measures extreme entries directly
- useful for detecting spikes or outlier parameters

**What can be learned**

- whether the layer is dominated by rare, very large weights
- whether optimization produces sharp local concentration in parameter space

### `rms`

**Definition**

- root mean square

$$
{rms}(W) = \sqrt{{mean}(W_{ij}^2)}
$$

**Example**

- mean square is `(1 + 4 + 4 + 0) / 4 = 2.25`
- `rms = 1.5`

**Why chosen**

- scale measure normalized by layer size
- closely related to variance/energy style thinking

**What can be learned**

- whether typical squared magnitude is increasing
- how “energetic” the average entry is independent of parameter count

### `n_params`

**Definition**

- number of entries in the matrix

**Example**

- for a `2 x 2` matrix, `n_params = 4`

**Why chosen**

- metadata useful for interpreting other statistics
- some downstream analyses may want to normalize or stratify by size

**What can be learned**

- not a dynamic property by itself in fixed architectures
- useful to understand whether a summary comes from a very large or very small layer

### `n_rows`

**Definition**

- number of rows after reshaping

**Why chosen**

- exposes output dimension or leading structural dimension

**What can be learned**

- architecture structure, especially when comparing layers or reshaped convolutional kernels

### `n_cols`

**Definition**

- number of columns after reshaping

**Why chosen**

- exposes effective input-side dimension after flattening

**What can be learned**

- structural context for the other matrix metrics

### `aspect_ratio`

**Definition**

$$
{aspect\_ratio}(W) =
\frac{\max(n_{\mathrm{rows}}, n_{\mathrm{cols}})}
{\min(n_{\mathrm{rows}}, n_{\mathrm{cols}}) + \varepsilon}
$$

**Example**

- for a `2 x 8` matrix, `aspect_ratio = 8 / 2 = 4`

**Why chosen**

- distinguishes tall, square, and wide layers
- matrix shape often changes how norms and singular-value-like behavior should be interpreted

**What can be learned**

- whether a metric comes from a highly anisotropic layer
- whether certain model families rely on very rectangular operators

### `row_l2_mean`

**Definition**

- compute the `l2` norm of each row
- average those row norms

$$
{row\_l2\_mean}(W) =
\frac{1}{n_{\mathrm{rows}}}
\sum_i \sqrt{\sum_j W_{ij}^2}
$$

**Example**

- row norms of the example matrix are:
  - row 1: `sqrt(1^2 + (-2)^2) = sqrt(5)`
  - row 2: `sqrt(2^2 + 0^2) = 2`
- `row_l2_mean = (sqrt(5) + 2) / 2`

**Why chosen**

- measures average row strength
- useful when rows correspond to output units, filters, or channels

**What can be learned**

- whether activity is broadly distributed across output-side directions

### `row_l2_max`

**Definition**

- maximum row `l2` norm

$$
{row\_l2\_max}(W) =
\max_i \sqrt{\sum_j W_{ij}^2}
$$

**Why chosen**

- detects the strongest row/output direction

**What can be learned**

- whether one output direction/filter dominates the layer

### `col_l2_mean`

**Definition**

- compute `l2` norm of each column
- average them

$$
{col\_l2\_mean}(W) =
\frac{1}{n_{\mathrm{cols}}}
\sum_j \sqrt{\sum_i W_{ij}^2}
$$

**Why chosen**

- input-side analogue of `row_l2_mean`

**What can be learned**

- whether strength is distributed across input dimensions

### `col_l2_max`

**Definition**

- maximum column `l2` norm

$$
{col\_l2\_max}(W) =
\max_j \sqrt{\sum_i W_{ij}^2}
$$

**Why chosen**

- detects dominant input-side directions

**What can be learned**

- whether the layer relies very heavily on a small subset of input directions

---

## Metric Family: `sharpness`

Important: these are not true loss-landscape sharpness metrics. They are cheap weight-space proxies.

They were chosen because they can be computed from checkpoints alone, without gradients, Hessians, or data batches.

### `sharp_proxy_maxabs_over_l2`

**Definition**

$$
\frac{{max\_abs}}{\|x\|_2 + \varepsilon}
$$

**Interpretation**

- compares the largest single entry to total Euclidean scale

**Example**

- if `max_abs = 10` and `l2 = 20`
- value = `0.5`

**Why chosen**

- detects whether one coordinate is unusually dominant relative to total norm

**What can be learned**

- whether a layer is becoming spike-dominated instead of broadly distributed

### `sharp_proxy_maxabs_over_meanabs`

**Definition**

$$
\frac{{max\_abs}}{{mean\_abs} + \varepsilon}
$$

**Interpretation**

- how extreme the largest entry is compared to the typical absolute entry

**Example**

- if `max_abs = 10` and `mean_abs = 2`
- value = `5`

**Why chosen**

- direct concentration/outlier measure

**What can be learned**

- whether the layer develops rare but very large weights

### `sharp_proxy_rms_over_meanabs`

**Definition**

$$
\frac{{rms}}{{mean\_abs} + \varepsilon}
$$

**Interpretation**

- compares quadratic scale to absolute-average scale

**Why chosen**

- sensitive to heavy tails and uneven magnitude distributions

**What can be learned**

- whether the distribution of weights becomes more dispersed or heavy-tailed

### `sharp_proxy_row_imbalance`

**Definition**

$$
\frac{{row\_l2\_max}}{{row\_l2\_mean} + \varepsilon}
$$

**Interpretation**

- how much the strongest row dominates the average row

**Example**

- if row norms are `[2, 2, 10]`
- `row_l2_mean = 14/3`
- ratio is a bit above `2`

**Why chosen**

- captures output-side imbalance

**What can be learned**

- whether one neuron/filter/output channel becomes much stronger than the rest

### `sharp_proxy_col_imbalance`

**Definition**

$$
\frac{{col\_l2\_max}}{{col\_l2\_mean} + \varepsilon}
$$

**Interpretation**

- input-side version of row imbalance

**Why chosen**

- captures dominance on the input dimension side

**What can be learned**

- whether the layer relies disproportionately on a small number of input directions

---

## Metric Family: `distance`

These metrics are model-level only. They compare full checkpoint parameter vectors over time.

Let:

- `theta_t` be the full flattened parameter vector at epoch `t`
- `theta_0` be the first checkpoint for that model
- `theta_{t-1}` be the previous checkpoint

These metrics were chosen because they describe optimization trajectory, not just static parameter shape.

### `distance_to_init_l2`

**Definition**

$$
\|\theta_t - \theta_0\|_2
$$

**Example**

- if `theta_0 = [0, 0]` and `theta_t = [3, 4]`
- value = `5`

**Why chosen**

- classic measure of how far training moved the model from initialization

**What can be learned**

- whether better models tend to move farther or stay closer to initialization

### `relative_distance_to_init_l2`

**Definition**

$$
\frac{\|\theta_t - \theta_0\|_2}{\|\theta_0\|_2 + \varepsilon}
$$

**Why chosen**

- normalizes distance by initial scale
- more comparable when models have different initialization magnitudes

**What can be learned**

- whether training movement is large relative to where the model started

### `distance_to_prev_l2`

**Definition**

$$
\|\theta_t - \theta_{t-1}\|_2
$$

**Why chosen**

- local step-size proxy in parameter space

**What can be learned**

- whether the optimizer is making large structural changes at a given epoch
- whether instability or cross-over events coincide with large parameter moves

### `cosine_to_init`

**Definition**

- cosine similarity between `theta_t` and `theta_0`

**Formula**

$$
\frac{\theta_t \cdot \theta_0}{\|\theta_t\|_2 \|\theta_0\|_2 + \varepsilon}
$$

**Example**

- if the vectors point in the same direction, value is near `1`
- if orthogonal, near `0`
- if opposite, near `-1`

**Why chosen**

- distance alone does not say whether the model changed directionally or mostly scaled up/down

**What can be learned**

- whether the model remains aligned with initialization or rotates away from it

### `cosine_to_prev`

**Definition**

- cosine similarity between `theta_t` and `theta_{t-1}`

$$
\frac{\theta_t \cdot \theta_{t-1}}{\|\theta_t\|_2 \|\theta_{t-1}\|_2 + \varepsilon}
$$

**Why chosen**

- captures directional continuity between consecutive checkpoints

**What can be learned**

- whether training proceeds smoothly or with abrupt directional changes

### `cumulative_path_length_l2`

**Definition**

- sum over epochs of `distance_to_prev_l2`

**Formula**

$$
\sum_{k=1}^{t} \|\theta_k - \theta_{k-1}\|_2
$$

**Example**

- if the model moves 2 units, then 3 units, cumulative path is `5`

**Why chosen**

- distinguishes total travel distance from net displacement

**What can be learned**

- whether optimization wandered a lot even if final displacement is modest

### `straightness_ratio`

**Definition**

$$
\frac{\|\theta_t - \theta_0\|_2}{{cumulative\_path\_length\_l2} + \varepsilon}
$$

**Range in practice**

- near `1`: movement is straight/direct
- much smaller than `1`: movement is indirect/wandering

**Example**

- if net displacement is `5` and cumulative path is `10`
- ratio = `0.5`

**Why chosen**

- summarizes efficiency/directness of the optimization path

**What can be learned**

- whether successful models follow a direct route in parameter space or a long meandering one

---

## Scope Semantics

### `weight_only`

Use only the main weight tensor of a layer.

**Why important**

- isolates the dominant linear operator in the layer
- avoids conflating weight dynamics with bias dynamics

### `combined`

Combine weight and bias before computing model-level norms.

**How**

- for `l1`

$$
\|[w; b]\|_1 = \|w\|_1 + \|b\|_1
$$

- for `l2`

$$
\|[w; b]\|_2 = \sqrt{\|w\|_2^2 + \|b\|_2^2}
$$

- for `linf`

$$
\|[w; b]\|_\infty = \max(\|w\|_\infty, \|b\|_\infty)
$$

**Why important**

- reflects the full parameter group instead of only the kernel/weight matrix

### `default`

Used for metrics where the current implementation does not provide separate weight-only vs combined variants.

---

## Why These Metrics Were Chosen

The design is pragmatic:

- checkpoint-only: no extra data batches required
- cheap: can be computed for many models and many epochs
- architecture-aware but still general enough for mixed model zoos
- complementary:
  - norms capture scale
  - matrix metrics capture shape and concentration
  - sharpness proxies capture imbalance/spikiness
  - distance metrics capture optimization trajectory

This makes them good candidates for exploratory analyses such as:

- weight-statistic vs performance-rank correspondence
- relation to ranking flips or cross-over events
- dataset-by-dataset comparisons of training dynamics

---

## How To Read These Metrics In Practice

Some useful heuristics:

- **Rising `l2` or `fro_norm`:**
  - parameter energy is growing
  - may indicate stronger separation, overfitting, or simply scale growth depending on context

- **Rising `max_abs` or `sharp_proxy_maxabs_over_meanabs`:**
  - more extreme coordinates are emerging
  - often suggests increasing concentration in a few parameters

- **High `row_imbalance` or `col_imbalance`:**
  - one row/column direction dominates
  - may indicate specialization or fragile dependence on a few directions

- **Large `distance_to_prev_l2` spikes:**
  - optimization is making abrupt moves
  - worth comparing to functional rank flips

- **High `straightness_ratio`:**
  - the optimization path is direct

- **Low `straightness_ratio`:**
  - the model wandered substantially before reaching its current state

---

## Important Caveats

- The `sharpness` family is only a proxy family. It is not Hessian sharpness, SAM sharpness, or loss-surface curvature.
- `n_params`, `n_rows`, and `n_cols` are mostly structural descriptors. In a fixed architecture they may not change over time.
- Matrix reshaping is a useful approximation, but it is still an approximation for high-dimensional tensors.
- A statistic being correlated with performance does not mean it is causal.

---

## Source

Implementation source:

- `learning-dynamics/src/nnrun/compute/weight_statistics_compute.py`