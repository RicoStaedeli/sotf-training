# Latent Hierarchical Transformer Regressor

A two-stage hierarchical transformer that reads variable-length sequences of latent embeddings
(one 2-D array per training sample) and regresses a scalar target — typically a future
checkpoint performance metric such as test accuracy.

---

## Table of Contents

1. [Conceptual Motivation](#1-conceptual-motivation)
2. [End-to-End Data Flow](#2-end-to-end-data-flow)
3. [Stage 0 — Raw DataFrame Input](#3-stage-0--raw-dataframe-input)
4. [Stage 1 — Preprocessing and Scaling](#4-stage-1--preprocessing-and-scaling)
5. [Stage 2 — DataLoader and Collation](#5-stage-2--dataloader-and-collation)
6. [Stage 3 — Network Forward Pass](#6-stage-3--network-forward-pass)
   - [3a Input Projection](#3a-input-projection)
   - [3b Local Stage (chunked self-attention)](#3b-local-stage-chunked-self-attention)
   - [3c Global Stage (CLS aggregation)](#3c-global-stage-cls-aggregation)
   - [3d Regression Head](#3d-regression-head)
7. [Flat Mode (local_window_size = 1)](#7-flat-mode-local_window_size--1)
8. [Positional Encoding](#8-positional-encoding)
9. [Architecture Diagram](#9-architecture-diagram)
10. [Configuration Reference](#10-configuration-reference)
11. [Serialisation / Pickle](#11-serialisation--pickle)
12. [Plain-Language Explainer: Windows, Padding, and Positional Encoding](#12-plain-language-explainer-windows-padding-and-positional-encoding)

---

## 1. Conceptual Motivation

Each training sample represents one *checkpoint observation* of a neural-network model during
training. Its features are not a flat vector but a **sequence of window embeddings**: a 2-D
array of shape `[T, D_in]` where:

- `T` = number of time-steps (epochs or sliding windows) seen up to this checkpoint.  
  Different models have different training histories, so `T` varies across samples.
- `D_in` = dimension of the latent embedding at each step (fixed across all samples for a
  given dataset).

A plain transformer on the full sequence of length `T` has O(T²) attention cost. The
**hierarchical design** reduces this by first compressing every non-overlapping chunk of
`local_window_size` tokens into one embedding (Stage 1 = *local encoder*), then running a
second transformer over the much shorter sequence of chunk-level embeddings (Stage 2 =
*global encoder*). The global encoder therefore sees at most `ceil(T / local_window_size)`
positions regardless of how long the original sequence is.

When `local_window_size = 1` the local stage is skipped entirely and the model degrades
gracefully to a standard flat transformer (full sequence → global encoder directly).

---

## 2. End-to-End Data Flow

```
DataFrame X (one row = one checkpoint)
│   column "window_sequence"  → list of 2-D arrays  shape [T_i, D_in]
│   other numeric columns     → scalar side-channel  shape [N, S]
│
├─ extract_window_sequence_features()
│     separates sequence column from scalar columns
│
├─ _truncate_sequences()          (optional, max_seq_len)
│
├─ fit/apply StandardScaler       per-feature mean/std on sequences
│   fit/apply StandardScaler       per-column on scalar features
│   fit/apply TargetScaler         mean/std on y
│
├─ build_window_sequence_dataloaders()
│     train/val split → WindowSequenceDataset → DataLoader
│           └─ collate_window_sequences()
│                 pad_sequence → "sequences" [B, T_max, D_in]
│                                "lengths"   [B]
│                                "scalar_features" [B, S]
│
└─ LatentHierarchicalTransformerNetwork.forward()
      │
      ├─ input_proj  Linear(D_in → d_model)        [B, T, d_model]
      │
      ├─ [if local_window_size > 1]  _local_stage()
      │     pad to multiple of lws
      │     reshape [B, T', d_model] → [B*n_win, lws, d_model]
      │     local_pos_enc (sinusoidal, resets per chunk)
      │     local_encoder (TransformerEncoder, num_local_layers)
      │     mean-pool → [B*n_win, d_model]
      │     reshape → [B, n_win, d_model]
      │     build window_mask [B, n_win]
      │
      ├─ _global_stage()
      │     prepend [CLS] → [B, 1+n_win, d_model]
      │     global_pos_enc (sinusoidal)
      │     full padding mask: CLS always visible, pad windows masked
      │     global_encoder (TransformerEncoder, num_global_layers)
      │     extract CLS[:,0,:]  → [B, d_model]
      │     final LayerNorm
      │
      ├─ concat scalar_features → [B, d_model + S]
      │
      └─ head (MLP)  → [B, 1]
            └─ invert_target_scaler → [N] float64
```

---

## 3. Stage 0 — Raw DataFrame Input

`fit(X, y)` and `predict(X)` receive a **pandas DataFrame** `X` with at least one special
column (default name: `"window_sequence"`) that holds the latent sequence for each sample.

### The sequence column

Each cell in the sequence column contains a 2-D NumPy array:

```
X["window_sequence"][i]  →  np.ndarray, shape [T_i, D_in]
```

- **`T_i`**: number of observed checkpoints / window embeddings for sample `i`.  
  Can differ between samples (variable-length). All must be non-zero and finite.
- **`D_in`**: embedding dimension, identical for every sample in the dataset.

`extract_window_sequence_features()` in `data/torch_datasets.py`:

1. Pulls the sequence column out, validates every entry (2-D, non-empty, finite), and
   returns it as `list[np.ndarray]`.
2. Drops the sequence column from `X`, converts the remainder to `float32`; this becomes
   the **scalar feature matrix** of shape `[N, S]` where `S` is the number of remaining
   columns (can be 0).

### Scalar features (side-channel)

Columns that survive after dropping the sequence column are numeric scalars per sample.
In the latent-window forecasting context these are optional temporal metadata features
(epoch number, history length, progress fraction, current target value, etc.) constructed
by `_latent_window_scalar_features()` in `forecasting/dataset.py`.

---

## 4. Stage 1 — Preprocessing and Scaling

Three independent scalers are fit once during `fit()` and stored on the regressor:

| Scaler | Applied to | Purpose |
|---|---|---|
| `_sequence_scaler` (StandardScaler) | All tokens across all sequences (stacked into `[Σ T_i, D_in]`) | Zero-mean, unit-variance per embedding dimension |
| `_scalar_scaler` (StandardScaler) | Scalar feature matrix `[N, S]` | Zero-mean, unit-variance per scalar column |
| `_target_scaler` (TargetScaler) | Regression target `y` | Mean-subtracted, divided by std |

During `predict()` the same fitted scalers are reused; `invert_target_scaler` is applied to
network output to recover the original scale.

If `scale_features=False` or `scale_target=False` the corresponding scaler is a no-op
identity transform (mean=0, scale=1).

**Optional truncation** (`max_seq_len`): Before scaling, each sequence array `[T_i, D_in]`
is sliced to `[:max_seq_len]` to cap the maximum sequence length seen by the network. This
is applied identically at fit and predict time.

---

## 5. Stage 2 — DataLoader and Collation

### Dataset

`WindowSequenceDataset` stores:
- `sequences`: list of `torch.FloatTensor` each of shape `[T_i, D_in]`
- `scalar_features`: `torch.FloatTensor [N, S]`
- `targets`: `torch.FloatTensor [N, 1]`

A `(features_dict, target)` tuple is returned per item, where `features_dict` has keys
`"sequence"` and `"scalar_features"`.

### Collation (`collate_window_sequences`)

Since sequences have different lengths within a batch, `collate_window_sequences` pads them:

1. Gathers `[T_i, D_in]` tensors from the batch.
2. Records their original lengths as integer tensor `lengths [B]`.
3. Calls `torch.nn.utils.rnn.pad_sequence(..., batch_first=True)` → **zero-padded** tensor
   `sequences [B, T_max, D_in]` where `T_max = max T_i` in the batch.
4. Stacks scalar features → `[B, S]`.

The resulting batch dict passed to the network is:

```python
{
    "sequences":       FloatTensor  [B, T_max, D_in],   # zero-padded
    "lengths":         LongTensor   [B],                 # true lengths
    "scalar_features": FloatTensor  [B, S],              # may have S=0
}
```

### Train/validation split

`split_train_validation_indices()` randomly holds out `validation_fraction` (default 15%)
of samples as a validation set for early stopping. The split is fixed by `random_state`.

---

## 6. Stage 3 — Network Forward Pass

### 3a Input Projection

```
sequences  [B, T_max, D_in]
     │
     └─ nn.Linear(D_in, d_model)
     │
     ▼
x  [B, T_max, d_model]
```

All input tokens are projected from their raw embedding dimension `D_in` to the internal
model dimension `d_model`. This is the only place `D_in` matters; the rest of the network
operates entirely in `d_model`-dimensional space.

---

### 3b Local Stage (chunked self-attention)

*Active only when `local_window_size > 1` and `num_local_layers > 0`.*

**Goal**: process each non-overlapping chunk of `local_window_size` consecutive tokens with
a shared transformer, then average-pool the chunk down to one vector.

#### Step-by-step

**1. Pad to multiple of `lws`**

```
T_max  possibly not divisible by lws
pad_len = (lws - T_max % lws) % lws
x padded to [B, T_max + pad_len, d_model]  (zero-padding on the right)
padded_len = T_max + pad_len
n_windows  = padded_len // lws
```

**2. Reshape into independent chunks**

```
[B, padded_len, d_model]
  →  view  →
[B * n_windows, lws, d_model]
```

Each chunk is now an independent mini-sequence of exactly `lws` tokens. The batch dimension
is `B * n_windows`; all chunks from all samples in the batch are processed simultaneously
by the **shared** local encoder.

**3. Local positional encoding** (if enabled)

Sinusoidal PE is added to the `[B*n_windows, lws, d_model]` tensor. Positions 0…lws−1 are
assigned to every chunk independently — the encoding resets at the beginning of each chunk,
so tokens learn their within-chunk position, not their global position.

**4. Local self-attention**

```
x  [B * n_windows, lws, d_model]
     │
     └─ TransformerEncoder (num_local_layers × TransformerEncoderLayer)
     ▼
x  [B * n_windows, lws, d_model]
```

Each layer applies: Pre-LayerNorm → multi-head self-attention (nhead heads) → residual →
Pre-LayerNorm → feed-forward (d_model → dim_feedforward → d_model) → residual.

No padding mask is passed here: within a chunk all `lws` positions are always valid
because the zero-padding appended in step 1 only ever falls in the **last chunk** of
sequences whose length is not a multiple of `lws`. This is a deliberate simplification —
the global-stage mask will ignore fully-padded windows anyway.

**5. Mean pooling**

```
x.mean(dim=1)   →   [B * n_windows, d_model]
```

All `lws` token representations within a chunk are averaged into one *window embedding*.

**6. Reshape back**

```
view →  [B, n_windows, d_model]
```

**7. Build window-level padding mask**

```
num_real_windows[i] = ceil(lengths[i] / lws)   # real windows for sample i
win_positions        = [0, 1, …, n_windows-1]   # broadcasted over batch
window_mask[i, j]   = True  if j >= num_real_windows[i]   (pad window)
                    = False if j <  num_real_windows[i]    (real window)
shape:  [B, n_windows]
```

This mask is passed to the global encoder so attention ignores positions that came from
zero-padded tokens.

---

### 3c Global Stage (CLS aggregation)

**Goal**: aggregate the `n_windows` window embeddings into one fixed-size vector using a
learnable `[CLS]` token, matching the BERT-style classification head pattern.

**1. Prepend CLS token**

```
cls_token  [1, 1, d_model]  (learnable parameter, trunc_normal init σ=0.02)
expand  →  [B, 1, d_model]
cat(cls, x)  →  [B, 1 + n_windows, d_model]
```

**2. Global positional encoding** (if enabled)

Sinusoidal PE over positions 0…n_windows (including position 0 for CLS). This encodes the
*inter-chunk* ordering: window 0 comes from the earliest part of the sequence, window
`n_windows-1` from the latest.

**3. Full padding mask**

```
cls_mask    [B, 1]              all False (CLS is never masked)
full_mask   cat([cls_mask, window_mask], dim=1)  →  [B, 1 + n_windows]
```

The `[CLS]` position is always attended to; padding windows are masked out.

**4. Global self-attention**

```
x  [B, 1 + n_windows, d_model]
     │
     └─ TransformerEncoder(
            num_global_layers × TransformerEncoderLayer,
            src_key_padding_mask = full_mask
        )
     ▼
x  [B, 1 + n_windows, d_model]
```

Identical layer structure to the local encoder (Pre-LN, nhead heads, dim_feedforward FFN)
but now with the padding mask. The CLS token can attend to all real window positions; real
window tokens can attend to each other and to CLS; pad positions are ignored everywhere.

**5. Extract CLS and normalise**

```
cls_out = x[:, 0, :]          →  [B, d_model]
cls_out = LayerNorm(cls_out)   →  [B, d_model]
```

The CLS vector at index 0 is the sequence-level representation. The final LayerNorm
(`self.final_norm`) stabilises magnitudes before feeding the head.

---

### 3d Regression Head

```
scalar_features  [B, S]

if S > 0:
    cat([cls_out, scalar_features], dim=1)  →  [B, d_model + S]
else:
    cls_out  →  [B, d_model]

MLP:
    Linear(d_model + S, head_hidden[0])
    activation  (GELU by default)
    Dropout(dropout)
    …  (one block per entry in head_hidden_layer_sizes)
    Linear(head_hidden[-1], 1)

output  →  [B, 1]   (scaled prediction)
```

The scalar side-channel features (e.g. current epoch, history length) are concatenated
directly with the CLS representation, bypassing the transformer entirely. This lets the
head exploit simple linear relationships between temporal metadata and the target without
having to learn them from attention patterns.

After `predict()`, the output is inverse-scaled by `invert_target_scaler` to recover the
original metric range, then returned as a 1-D float64 array of shape `[N]`.

---

## 7. Flat Mode (`local_window_size = 1`)

When `local_window_size == 1` (or `num_local_layers == 0`) the local encoder is skipped.
The projected tokens `[B, T_max, d_model]` are passed **directly** to the global stage with
no chunking. The positional encoding is still applied before the global encoder (local_pos_enc
is reused as the token-level PE in this path). The global padding mask is built directly
from `lengths`:

```
window_mask[i, j] = True  if j >= lengths[i]
```

This makes the model behave as a standard BERT-style transformer encoder where each token
is a raw projected input.

---

## 8. Positional Encoding

`_SinusoidalPositionalEncoding` implements the fixed sinusoidal scheme from Vaswani et al.
(2017). It pre-computes a table `pe [max_len, d_model]` and caches it as a non-trainable
buffer:

```
pe[pos, 2i]   = sin(pos / 10000^(2i/d_model))
pe[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
```

The even/odd formula handles both even and odd `d_model` correctly by computing
`n_even = ceil(d_model/2)` independent frequency terms and assigning them to alternating
slots. At forward time `pe[:seq_len]` is broadcast-added over the batch dimension.

Two independent PE modules exist:
- `local_pos_enc`: added within each chunk (positions 0…lws−1, reset per chunk; in flat
  mode used for the full token sequence).
- `global_pos_enc`: added over the global sequence of window embeddings plus CLS
  (positions 0…n_windows).

Set `use_positional_encoding=False` to disable both (both become `nn.Identity`).

---

## 9. Architecture Diagram

```
Input DataFrame X
┌──────────────────────────────────────────────────────────┐
│  "window_sequence"  │  scalar_feat_1  │  …  │  scalar_S  │
│  [T_i, D_in]        │  float          │     │  float     │
└──────────────────────────────────────────────────────────┘
           │                       │
           ▼                       ▼
  Standard Scaler            Standard Scaler
           │                       │
           ▼                       │
  pad_sequence collation           │
  [B, T_max, D_in]                 │
           │                       │
           ▼                       │
  Linear(D_in → d_model)           │
  [B, T_max, d_model]              │
           │                       │
    ┌──────▼──────┐                │
    │  LOCAL STAGE │  (lws > 1)    │
    │              │               │
    │  reshape to  │               │
    │  [B*n, lws,  │               │
    │   d_model]   │               │
    │  + local PE  │               │
    │  TransfEnc   │               │
    │  mean-pool   │               │
    │  [B,n,d]     │               │
    └──────┬───────┘               │
           │                       │
    ┌──────▼────────────┐          │
    │   GLOBAL STAGE    │          │
    │                   │          │
    │  [CLS] prepended  │          │
    │  + global PE      │          │
    │  TransfEnc        │          │
    │  + padding mask   │          │
    │  extract CLS[:,0] │          │
    │  LayerNorm        │          │
    │  [B, d_model]     │          │
    └──────┬────────────┘          │
           │                       │
           └──────────┬────────────┘
                      │  cat  [B, d_model + S]
                      ▼
              ┌──────────────┐
              │ MLP Head     │
              │ Linear→GELU  │
              │ …            │
              │ Linear→ [B,1]│
              └──────┬───────┘
                     │  invert_target_scaler
                     ▼
              predictions  [N] float64
```

---

## 10. Configuration Reference

All parameters are set through `LatentHierarchicalTransformerConfig` (or equivalently as
keyword arguments to `LatentHierarchicalTransformerRegressor`).

### Architecture

| Parameter | Default | Description |
|---|---|---|
| `d_model` | `128` | Internal transformer dimension. Input tokens and all attention computations live in this space. |
| `nhead` | `4` | Number of attention heads. Must divide `d_model`. |
| `num_local_layers` | `2` | Depth of the local encoder. Set to 0 (or `local_window_size=1`) to skip the local stage entirely. |
| `num_global_layers` | `2` | Depth of the global encoder. Must be ≥ 1. |
| `local_window_size` | `1` | Chunk size for the local stage. `1` = flat single-stage mode. |
| `dim_feedforward` | `256` | Hidden size of the FFN inside each transformer layer. |
| `dropout` | `0.1` | Dropout applied inside attention, FFN, and the MLP head. |
| `head_hidden_layer_sizes` | `(64,)` | Hidden widths of the regression head MLP. Empty tuple = single linear layer. |
| `activation` | `"gelu"` | Head activation: `relu`, `gelu`, `silu`, `elu`, `leaky_relu`. |
| `use_positional_encoding` | `True` | Whether to add sinusoidal PE in both stages. |

### Data

| Parameter | Default | Description |
|---|---|---|
| `sequence_column` | `"window_sequence"` | DataFrame column holding the latent sequence arrays. |
| `max_seq_len` | `None` | Truncate sequences to this many tokens before any processing. |

### Training

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | `16` | Mini-batch size. `"auto"` → clamp to [16, 256] based on dataset size. |
| `learning_rate` | `1e-3` | Adam initial learning rate. Alias: `learning_rate_init`. |
| `weight_decay` | `1e-3` | AdamW L2 penalty. Alias: `alpha`. |
| `max_epochs` | `500` | Maximum training epochs. Alias: `max_iter`. |
| `patience` | `40` | Early-stopping patience (epochs without val-loss improvement). |
| `early_stopping` | `True` | Whether to use early stopping. |
| `scheduler_factor` | `0.5` | ReduceLROnPlateau multiplicative factor. |
| `scheduler_patience` | `15` | Epochs before LR is reduced. |
| `min_learning_rate` | `1e-5` | LR floor for the scheduler. |
| `gradient_clip_norm` | `1.0` | Max gradient norm for clipping (`None` = disabled). |
| `scale_features` | `True` | Whether to standardise input features. |
| `scale_target` | `True` | Whether to standardise the regression target. |
| `loss` | `"smooth_l1"` | Training loss: `"smooth_l1"` (Huber) or `"mse"`. |
| `validation_fraction` | `0.15` | Fraction of training data held out for validation / early stopping. |
| `random_state` | `42` | Seed for train/val split and weight init. |
| `device` | `"auto"` | `"auto"` selects CUDA/MPS/CPU; or explicit `"cpu"`, `"cuda"`, `"mps"`. |

---

## 11. Serialisation / Pickle

The regressor is designed to survive `pickle.dump` / `pickle.load` across different devices
(e.g. trained on CUDA, loaded on CPU).

`__getstate__` removes the live `_network` module and replaces it with a CPU copy of its
`state_dict`. `__setstate__` re-creates the network architecture from the stored
`_input_dim` / `_scalar_dim` / config, loads the weights, and re-resolves the device with
`resolve_device("auto")`. This makes `model_bundle.pkl` files fully portable.

---

## 12. Plain-Language Explainer: Windows, Padding, and Positional Encoding

This section explains the three concepts from first principles, without assuming any prior
knowledge of transformers.

---

### 12.1 What is a window and what does it represent?

Forget the code for a second. Imagine you are trying to predict how well a neural network
will perform at the end of training, and you are observing it at epoch 30 out of 100.

At epoch 30 you have a **history**: you have seen the network's weight statistics at epochs
1, 2, 3, … 30. Each of those checkpoints produced a latent embedding — a vector of numbers
summarising "what the network looks like at this point in training." So your input is a
sequence of 30 vectors, each of dimension `D_in`. That is the `[T=30, D_in]` array stored
in the `window_sequence` column.

The transformer needs to read this whole sequence and produce one number (the predicted
future accuracy). The problem: a transformer's attention is **quadratic in sequence
length**. If you have 300 checkpoints, attention needs to compute 300×300 = 90,000 pairs.
For long training histories that gets expensive fast.

**The window trick cuts that cost.** Instead of attending over all 30 tokens at once, you:

1. Split the sequence into non-overlapping chunks of size `local_window_size` (say, 5):
   - Chunk 0: epochs 1–5
   - Chunk 1: epochs 6–10
   - …
   - Chunk 5: epochs 26–30

2. Run a small transformer **inside each chunk** independently. Tokens in chunk 0 only
   talk to each other; tokens in chunk 1 only talk to each other. This is the *local
   encoder*.

3. Average-pool each chunk's 5 output vectors into **one vector** — the chunk's summary,
   called a *window embedding*.

4. Now instead of 30 tokens you have 6 window embeddings. Run a second transformer over
   those 6 embeddings — the *global encoder*. Attention is now 6×6 = 36 pairs instead of
   30×30 = 900.

**What does a window embedding represent?** It represents "what was happening to this
network during this short slice of training." The local transformer learns to pick up on
patterns within the window (e.g. "loss was dropping fast in these 5 epochs"), and the
mean-pool collapses that into a single summary vector. The global transformer then reasons
about how those short-term patterns connect across the whole training history.

---

### 12.2 Padding — why it is needed and how it works

#### Problem: sequences have different lengths

Every sample in a batch has a different `T_i`. Sample A might have 28 epochs of history,
sample B might have 35, sample C might have 12. A GPU tensor must be a rectangular block —
all rows the same length. You cannot stack jagged arrays directly.

**Solution: zero-padding.** Extend every sequence to `T_max = max(T_i)` in the batch by
appending rows of zeros. So sample A becomes `[35, D_in]` with 7 extra zero rows at the
end. This lets everything be stacked into one `[B, T_max, D_in]` tensor.

But now the network must be told which positions are real and which are fake zeros —
otherwise the transformer will attend to those fake zero tokens and get confused. This is
done with a **padding mask**: a boolean matrix of shape `[B, T_max]` where `True` means
"this position is padding, ignore it."

#### The two-level padding problem

The hierarchical design introduces a second padding challenge on top of the first.

**Level 1 — alignment padding** (inside `_local_stage`): after you have `T_max`, you need
to split it into chunks of exactly `local_window_size`. But `T_max` might not be divisible
by `lws`. If `T_max=28` and `lws=5`, you get 5 full chunks (25 tokens) plus 3 leftover
tokens. You cannot process a partial chunk, so the code pads the sequence to the next
multiple of `lws` (here: 30), adding 2 more zero vectors. Now `28 → 30`, and you get
exactly 6 chunks.

```
T_max = 28,  lws = 5
pad_len = (5 - 28%5) % 5 = (5 - 3) % 5 = 2
padded_len = 30,  n_windows = 6
```

This alignment padding is purely structural — it makes the reshape arithmetic work. The
zero-padding tokens end up in the last chunk and are averaged in with real tokens during
mean-pool, but the global-stage mask will discard that window if it is entirely padding.

**Level 2 — window mask** (passed to the global encoder): after mean-pooling you have
`[B, n_windows, d_model]`. But for sample A with `T_i=28` and `lws=5`, only
`ceil(28/5)=6` windows are real (all 6 in this case). For sample B with `T_i=12`, only
`ceil(12/5)=3` windows are real; windows 3, 4, 5 are padding and must be masked.

```
num_real_windows[i] = ceil(lengths[i] / lws)
window_mask[i, j]   = True   if j >= num_real_windows[i]   ← ignore
                    = False  if j <  num_real_windows[i]   ← attend
```

This mask is passed into the global TransformerEncoder as `src_key_padding_mask`. PyTorch
uses it to set those positions' attention scores to `-inf` before the softmax, effectively
zeroing out their contribution.

The CLS token gets its own mask column of all-`False` so it is always visible.

#### Visual example

```
Batch of 3 samples, lws=3, T_max=7

Sample A: T=7  ████████████████████░  (0 padding tokens)
Sample B: T=5  ██████████████░░░░░░░  (2 padding tokens)
Sample C: T=3  ████████░░░░░░░░░░░░░  (4 padding tokens)

After alignment-padding to multiple of 3 → T_max padded = 9:

Sample A: ████████████████████████░░  chunks: [AAA][AAA][A░░]  → 3 real windows
Sample B: ██████████████░░░░░░░░░░░░  chunks: [BBB][BB░][░░░]  → 2 real windows
Sample C: ████████░░░░░░░░░░░░░░░░░░  chunks: [CCC][░░░][░░░]  → 1 real window

Global window mask:
         win0   win1   win2
A:      False  False  False   (all 3 real)
B:      False  False   True   (win2 masked)
C:      False   True   True   (win1, win2 masked)
```

---

### 12.3 Positional Encoding

Without positional encoding, a transformer is **order-blind**. The attention mechanism
computes dot products between all pairs of tokens simultaneously — it has no built-in
notion of "this token comes before that one." If you shuffle the input, you get the same
output. For sequences where order matters (epoch 1, 2, 3 … carry time-ordering
information), this is a fatal flaw.

The solution: **add a unique signal to each position** so the model can figure out where
it is.

#### The sinusoidal formula

For a token at position `pos` (0-indexed), add a vector of the same dimension `d_model`:

```
encoding[pos, 2i]   = sin( pos / 10000^(2i / d_model) )
encoding[pos, 2i+1] = cos( pos / 10000^(2i / d_model) )
```

`i` runs from 0 to `d_model/2 - 1`. You are filling pairs of dimensions with a sine/cosine
at a particular *frequency*. The frequencies span a huge range:

- **Dimension pair 0**: very high frequency — the sine completes a full cycle every ~6
  positions. Good for distinguishing adjacent tokens.
- **Dimension pair `d_model/2`**: very low frequency — the sine barely changes over
  thousands of positions. Good for distinguishing tokens far apart.

Think of it like a clock with many hands, each spinning at a different speed. Position 0
and position 1 might look nearly identical on the slow hands, but differ sharply on the
fast hands. Position 0 and position 512 differ clearly on even the slowest hand.

**Why not learned embeddings?** Sinusoidal PE generalises to sequence lengths never seen
during training, which matters here because `T_i` varies at inference time. A table of
learned position embeddings would have no entry for position 300 if the training data only
had up to 200 checkpoints.

#### Where it is applied in this model

There are **two independent PE modules**, both with the same formula but applied to
different things:

**`local_pos_enc`** — applied inside each chunk, before the local transformer:

```
chunk shape: [B * n_windows, lws, d_model]
add PE positions 0, 1, 2, …, lws-1
```

The position index **resets to 0 at the start of every chunk**. The token at position 3 in
chunk 0 and the token at position 3 in chunk 2 both get the same PE. This makes sense: the
local encoder is learning "what position within a short window does this token occupy" — it
does not need to know which chunk it is in.

**`global_pos_enc`** — applied over the window embeddings + CLS, before the global
transformer:

```
sequence shape: [B, 1 + n_windows, d_model]
add PE positions 0, 1, 2, …, n_windows
                 ↑CLS                ↑last window
```

Here position 0 goes to CLS and positions 1…n_windows go to the window embeddings in
order. The global encoder therefore knows: "window 1 came from an earlier part of training
history than window 5." This is the inter-chunk ordering signal.

#### Concrete example with tiny numbers

Say `d_model=4`, so we have 2 dimension pairs:

```
pair 0 (dims 0,1):  frequency = 1/10000^(0/4) = 1.0   (fast)
pair 1 (dims 2,3):  frequency = 1/10000^(2/4) = 0.01  (slow)

pos=0:  [sin(0),   cos(0),   sin(0),    cos(0)   ]  = [ 0.000,  1.000,  0.000,  1.000]
pos=1:  [sin(1),   cos(1),   sin(0.01), cos(0.01)]  = [ 0.841,  0.540,  0.010,  1.000]
pos=2:  [sin(2),   cos(2),   sin(0.02), cos(0.02)]  = [ 0.909, -0.416,  0.020,  1.000]
pos=5:  [sin(5),   cos(5),   sin(0.05), cos(0.05)]  = [-0.959,  0.284,  0.050,  0.999]
```

Position 1 and 2 look very similar on the slow dimensions (2, 3) but differ clearly on
dimensions 0 and 1. Position 0 and position 5 differ on both pairs. This is how attention
can figure out relative ordering — the network learns to read these "fingerprints."
