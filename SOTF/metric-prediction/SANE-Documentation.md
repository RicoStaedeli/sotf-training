# SANE — Technical Documentation

**SANE** (Self-supervised Autoencoder for Neural network Embeddings) is a transformer-based autoencoder that learns fixed-dimensional latent representations of trained neural network checkpoints. Instead of treating a checkpoint as a raw flat vector of weights, SANE tokenises the weight space into structured windows, uses learned 3-D positional embeddings to encode architectural position, and trains with a combination of reconstruction and contrastive objectives.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [SANE Model Architecture](#2-sane-model-architecture)
3. [Position Embeddings](#3-position-embeddings)
4. [Data Preprocessing Pipeline](#4-data-preprocessing-pipeline)
   - [4.2 Data Splits and Splitting Strategies](#42-data-splits-and-splitting-strategies)
5. [Tokenisation](#5-tokenisation)
6. [Windowing](#6-windowing)
7. [Loss Functions](#7-loss-functions)
8. [Training Setup](#8-training-setup)
9. [Configuration System](#9-configuration-system)
10. [export_latent.py — What It Does and What It Generates](#10-export_latentpy)
11. [Full Data Flow: Raw Checkpoint → Latent Embedding](#11-full-data-flow)
12. [Shapes and Dimensions Reference](#12-shapes-and-dimensions-reference)

---

## 1. Project Structure

```
internal-sane/
├── config/                  # Hydra configuration files
│   ├── model/               # Encoder, decoder, projector, embedder configs
│   ├── data/                # Dataset, tokenizer, cache configs
│   ├── loss/                # Loss function configs
│   ├── optimizer/           # Optimiser configs
│   └── scheduler/           # LR-scheduler configs
├── scripts/
│   └── export_latent.py     # Encode a full zoo → latent .pt files
├── src/sane/
│   ├── model/
│   │   ├── autoencoder.py   # SANEAutoEncoder — main model class
│   │   ├── transformer.py   # SANETransformerEncoder
│   │   ├── position.py      # Learned / sinusoidal position embeddings
│   │   ├── projection.py    # AverageProjectionHead, MLPProjectionHead
│   │   └── translator.py    # MLPTranslator (encoder→decoder bridge)
│   ├── data/
│   │   └── datasets/
│   │       ├── zoo_dataset.py             # Checkpoint loader
│   │       ├── windowed_dataset.py        # Window sampler
│   │       └── cached_windowed_dataset.py # Disk-cached variant
│   ├── loss/
│   │   ├── loss.py            # SANELoss base + CompositeLoss
│   │   ├── reconstruction.py  # MSEReconstructionLoss
│   │   └── contrastive.py     # NTXentLoss
│   └── trainer/
│       ├── trainer.py              # SANETrainer
│       └── contrastive_trainer.py  # SANEContrastiveTrainer
├── run.py                   # Training entry point
└── README.md
```

---

## 2. SANE Model Architecture

**File:** [src/sane/model/autoencoder.py](src/sane/model/autoencoder.py)  
**Main class:** `SANEAutoEncoder(nn.Module)`

SANE is a seq2seq autoencoder whose input and output are **windows of neural-network weights** represented as token sequences. Its core components are assembled from a Hydra config and can be mixed and matched independently.

```
Input window
 (B, N, token_dim)
       │
       ▼
 ┌─────────────┐
 │   Encoder   │  SANETransformerEncoder
 └──────┬──────┘
        │  z  (B, N, latent_dim)
        │
   ┌────┴────┐
   │         │
   ▼         ▼
┌──────┐  ┌──────────┐
│Embed │  │Projector │  (optional, contrastive)
└──┬───┘  └────┬─────┘
   │z_emb      │z_p
   │(B, D)     │(B, D)
   │
   ▼ (optional)
┌────────────┐
│ Translator │  MLPTranslator (can reshape token sequence)
└──────┬─────┘
       │ z̃  (B, N, latent_dim)
       ▼
 ┌─────────────┐
 │   Decoder   │  SANETransformerEncoder
 └──────┬──────┘
        │ x̂  (B, N, token_dim)
        ▼
 Reconstructed window
```

### 2.1 Encoder

**Class:** `SANETransformerEncoder` (see [Section 2.3](#23-saneransformerencoder-internals))

| Parameter | Default |
|-----------|---------|
| `input_dim` | matches tokeniser output (e.g. 256) |
| `d_model` | 1024 |
| `num_layers` | 16 |
| `nhead` | 16 |
| `output_dim` (`latent_dim`) | 128 |

Output shape: `(B, N, latent_dim)`.

### 2.2 Decoder

By default `decoder: "symmetric"` — a mirror of the encoder with swapped input/output projections. Can also be configured independently.

- Input: `(B, N, latent_dim)` or `(B, N, translator.output_dim)` if a translator exists.
- Output: `(B, N, token_dim)` — the reconstruction of the original window.

### 2.3 SANETransformerEncoder Internals

**File:** [src/sane/model/transformer.py](src/sane/model/transformer.py)

```
(B, N, input_dim)
    │
    ├─ input_layer: Linear(input_dim → d_model)   [or Identity if equal]
    │
    ├─ position_embedding(x, pos)                 [adds 3-D position info]
    │
    ├─ transformer_encoder                        [configurable type]
    │     "pytorch":  nn.TransformerEncoder  (batch_first=True)
    │     "gpt":      GPTTransformerEncoder  (causal=False, bias=False)
    │
    └─ output_layer: Linear(d_model → output_dim) [or Identity if equal]
         │
    (B, N, output_dim)
```

### 2.4 Embedder

Compresses the per-token latent matrix `(B, N, latent_dim)` into a single vector per checkpoint.

| Class | Operation | Output shape |
|-------|-----------|--------------|
| `AverageProjectionHead` | `z.mean(dim=1)` | `(B, latent_dim)` |
| `MLPProjectionHead` | Flatten → MLP | `(B, projection_dim)` |

Default is `AverageProjectionHead`. This is the path used by `export_latent.py` to produce one embedding per checkpoint.

### 2.5 Projector (optional)

Same head types as the embedder but used **only during contrastive training**. Produces `z_p` for NT-Xent loss. Not used at inference time.

### 2.6 Translator (optional)

**File:** [src/sane/model/translator.py](src/sane/model/translator.py)  
**Class:** `MLPTranslator`

Sits between encoder and decoder. Useful when the encoder and decoder operate on different token-sequence lengths or dimensions. Synthesises positional encodings for the decoder if the token count changes.

- Input: `(B, N_enc, latent_dim)`
- Output: `(B, N_dec, output_dim)`

### 2.7 Forward Pass Methods

```python
# Full autoencoder (training)
output = model.forward(x, p, mask=None)
# Returns dict:
#   'x'       → (B, N, token_dim) — original input
#   'p'       → (B, N, 3)         — positions
#   'mask'    → (B, N)             — padding mask
#   'z'       → (B, N, latent_dim) — encoder output
#   'z_p'     → (B, latent_dim)    — projector output (if projector exists)
#   'z_tilde' → (B, N, latent_dim) — translator output (if translator exists)
#   'x_hat'   → (B, N, token_dim)  — reconstruction

# Encoder + Embedder only (used by export_latent.py)
z_emb = model.forward_embeddings(x, p)   # → (B, latent_dim)

# Encoder + Projector only
z_p   = model.forward_projector(x, p)    # → (B, latent_dim) or (B, proj_dim)
```

---

## 3. Position Embeddings

**File:** [src/sane/model/position.py](src/sane/model/position.py)

Weights in a neural network have a natural **3-D structural address**:

| Dimension | Meaning | Example range |
|-----------|---------|---------------|
| 0 | Layer index | `[0, num_layers)` |
| 1 | Sub-module index within layer | `[0, max_submodules)` |
| 2 | Parameter index within sub-module | `[0, max_params)` |

SANE encodes this address via the `pos` tensor of shape `(B, N, 3)` and fuses it into the token stream.

### 3.1 LearnedPositionEmbeddings (default)

Three separate embedding tables `pe1`, `pe2`, `pe3` — one per dimension — each of shape `(max_positions[d], d_model // 2)`.

```python
pos_emb1 = pe1(pos[:, :, 0])   # (B, N, d_model//2)  — layer
pos_emb2 = pe2(pos[:, :, 1])   # (B, N, d_model//2)  — submodule
pos_emb3 = pe3(pos[:, :, 2])   # (B, N, d_model//2)  — param index

# First half: sum layer + submodule; second half: param index only
pos_emb = torch.cat([pos_emb1 + pos_emb2, pos_emb3], dim=-1)  # (B, N, d_model)

output = x + pos_emb   # residual addition
```

Default `max_positions`: `[1024, 256, 256]` — supports networks with up to 1024 layers.

### 3.2 FunctionalSinusoidalPositionEmbeddings

Generates sinusoidal embeddings on the fly (no learnable parameters). Accepts `(B, N, num_pos_dims)` and sums sinusoidal contributions across dimensions. Used when `position_embedding: "sinusoidal"` in config.

---

## 4. Data Preprocessing Pipeline

```
Disk: model_zoo/
       model_1/epoch_10/checkpoints/state_dict.pt
       model_1/epoch_20/checkpoints/state_dict.pt
       ...
       model_N/...
           │
           ▼
       ZooDataset           — loads checkpoint files, filters NaN / outliers
           │
           ▼
       Tokenisation         — state_dict → flat token + position sequences
           │
           ▼
       WindowedDataset      — slices sequences into fixed-length windows
           │
           ▼
  CachedWindowedDataset     — optional disk cache (avoids re-tokenising)
           │
           ▼
       DataLoader           — batches windows for GPU training
```

### 4.1 ZooDataset

**File:** [src/sane/data/datasets/zoo_dataset.py](src/sane/data/datasets/zoo_dataset.py)

Loads neural-network checkpoints from a "model zoo" directory structure. Key options:

| Config key | Effect |
|------------|--------|
| `epoch_idx` | List of epoch indices to load (e.g. `[10, 20, 30]`) |
| `drop_nan` | Skip checkpoints containing NaN weights |
| `max_absolute_weight` | Drop checkpoints where any weight exceeds this value |
| `max_checkpoints` | Limit total number of loaded checkpoints |
| `augmentations.align` | Enable weight-matching permutation augmentations |
| `augmentations.num_permutations` | Number of permuted views to generate |

Returns: `Checkpoint` objects containing the `nn.Module` and metadata (model config, evaluation results).

### 4.2 Data Splits and Splitting Strategies

**File:** [src/sane/data/splitter.py](src/sane/data/splitter.py)

Splitting determines which checkpoints land in train, validation, and test. The correct choice matters because model zoos often contain many checkpoints of the same architecture or hyperparameter configuration — a naive random split would allow nearly-identical checkpoints to appear in both train and test, leaking information and inflating downstream evaluation metrics.

All splitters share the same abstract interface: they accept a `CheckpointsDataset` and a sequence of fractional ratios (summing to 1) and return one `torch.utils.data.Subset` per split. Splitting always happens at the **checkpoint level**, never at the window level, so all windows derived from a given checkpoint always end up in the same split.

The splitter is selected by name in the data config and passed as a string to `DatasetFactoryConfig`:

```yaml
# config/data/my_zoo.yaml
splitter: "ModelGroupSplitter"   # RandomSplitter | ModelGroupSplitter | HyperparameterGroupSplitter | ModelWidthGroupSplitter
seed: 42                         # forwarded to the splitter for reproducible shuffling
split:
  train: 0.70
  val:   0.15
  test:  0.15
```

#### Splitter Overview

| Splitter | Granularity | Leakage risk | Typical use case |
|----------|-------------|--------------|-----------------|
| `RandomSplitter` | individual checkpoint | High | Quick experiments only |
| `ModelGroupSplitter` | entire model (all epochs) | Low | Standard; prevents same model appearing in multiple splits |
| `HyperparameterGroupSplitter` | hyperparameter config (ignores seed) | Very low | Zoos with multiple seeds per config; tests true generalisation |
| `ModelWidthGroupSplitter` | model (all epochs) + stratified by width | Low, width-aware | Width-sweep zoos where every width must be seen at training time |

---

#### 4.2.1 RandomSplitter

**Class:** `RandomSplitter`

The simplest strategy: wraps PyTorch's `random_split` to divide checkpoints uniformly at random according to the given ratios.

```
All checkpoints ──► random shuffle ──► [train | val | test]
```

**When to use:** Only for quick sanity checks or when the zoo contains a single checkpoint per model with no structural overlap.

**Warning:** This splitter is explicitly documented as a bad default choice. Because it splits at the checkpoint level without regard for model identity, it can place checkpoints from the same model in both train and test. It can also split multiple seeds of the same hyperparameter configuration across sets, making them trivially predictable from each other.

---

#### 4.2.2 ModelGroupSplitter

**Class:** `ModelGroupSplitter(seed=42)`

Groups all checkpoint indices that share the same `model_id` into a single group, then assigns **entire model groups** to splits. No model ever appears in more than one split, regardless of how many epochs or checkpoints it produced.

```
Checkpoints ──► group by model_id ──► shuffle groups ──► assign N_train groups to train
                                                         assign N_val   groups to val
                                                         assign N_test  groups to test
```

**Algorithm:**
1. Iterate all checkpoints and build a `{model_id → [idx, ...]}` map.
2. Shuffle the list of model groups with the given seed.
3. Greedily assign groups to splits: each split receives `round(ratio × n_models)` groups; the last split absorbs any rounding remainder.

**Requires:** The dataset must implement `get_model_id(index) → str` (provided by `ZooDataset`).

**When to use:** The standard choice for most zoos. Prevents same-model leakage across splits while respecting the target ratios as closely as possible at the model-group level.

**Note on ratio accuracy:** Because splitting is at the model level, the actual checkpoint counts in each split depend on how many checkpoints each model contributed. If models have unequal numbers of epochs or checkpoints, the realized checkpoint ratios may differ slightly from the configured ratios.

---

#### 4.2.3 HyperparameterGroupSplitter

**Class:** `HyperparameterGroupSplitter(seed=42, exclude_keys=["seed"])`

Groups models not just by identity but by **hyperparameter configuration**. It reads each model's `params.json`, strips the keys in `exclude_keys` (default: `["seed"]`), and hashes the remaining config with MD5. All models that hash identically — i.e., same config, different random seeds — are treated as a single group and forced into the same split.

```
params.json ──► remove excluded keys ──► MD5 hash ──► group by hash
                                                       │
                                              shuffle hyperparam groups
                                                       │
                                              assign groups to splits
```

**Algorithm:**
1. For each model, compute `hyperparam_hash = MD5(sorted(params - exclude_keys))`.
2. Build a `{hash → [model_id, ...]}` map (hyperparam groups).
3. Shuffle hyperparam groups with the given seed.
4. Assign entire hyperparam groups to splits using the same greedy rounding as `ModelGroupSplitter`.

**When to use:** Zoos that contain multiple models trained with identical hyperparameters but different random seeds (a common setup for variance analysis). A downstream predictor should not be able to exploit the fact that two test-set models are near-identical to training models — this splitter ensures all seeds of a configuration land in the same split.

**Configuration example:**

```yaml
splitter: "HyperparameterGroupSplitter"
# To also exclude learning_rate from the grouping key:
# exclude_keys: ["seed", "lr"]   # passed via DatasetFactoryConfig
```

---

#### 4.2.4 ModelWidthGroupSplitter

**Class:** `ModelWidthGroupSplitter(seed=42)`

Extends model-group splitting with **width stratification**. It keeps all checkpoints of the same model together (like `ModelGroupSplitter`) but additionally guarantees that the **train split contains at least one model from every observed width bucket**. This is critical when training a downstream predictor over width-sweep zoos — without stratification, entire width values could be absent from the training set.

```
Models ──► extract width from params.json ──► bucket by width
                                              │
                           for each width bucket:
                             shuffle models within bucket
                             allocate counts to splits (rounding-safe)
                             ensure train always gets ≥ 1 model
                                              │
                                     merge indices per split
```

**Width extraction:** Width is resolved in order of priority:
1. Top-level `model_width` or `width` key in `params.json` (searched recursively through nested dicts and lists).
2. Pattern match on the model directory path: `model_width=<value>` or `width=<value>`.
3. `ValueError` if neither is found.

**Allocation algorithm (`_allocate_counts`):** For each width bucket of size `n`, the desired counts per split are `[ratio_i × n]`. Counts are computed with integer truncation and the remainder is distributed to the splits with the largest fractional parts. If the train split ends up with zero models (possible with very small buckets), one model is transferred from the largest other split to guarantee coverage.

**When to use:** Width-sweep zoos (e.g. ResNet with `model_width ∈ {16, 32, 64, 128, 256}`) where a downstream task needs to generalise across widths. Ensures the SANE encoder is not evaluated on widths it has never seen during downstream training.

**Example zoo structure this targets:**

```
zoo/
  width=16_seed=1/   ──► width bucket "16"
  width=16_seed=2/   ──► width bucket "16"
  width=32_seed=1/   ──► width bucket "32"
  width=64_seed=1/   ──► width bucket "64"
  ...
```

With `ratios=[0.7, 0.15, 0.15]` and 2 models per width, the train split will receive at least one model from widths 16, 32, and 64.

---

#### Choosing a Splitter

```
Is the zoo a width sweep (different model_width values)?
  └─ YES ──► ModelWidthGroupSplitter
  └─ NO
      Does the zoo contain multiple seeds per hyperparameter config?
        └─ YES ──► HyperparameterGroupSplitter
        └─ NO
            Single checkpoint or quick test?
              └─ YES ──► RandomSplitter  (be aware of leakage)
              └─ NO  ──► ModelGroupSplitter  (safe default)
```

---

## 5. Tokenisation

The tokeniser converts a `state_dict` (ordered dict of weight tensors) into three parallel sequences:

| Output | Shape | Description |
|--------|-------|-------------|
| `tokens` | `(T, token_dim)` | Scalar weight values (token_dim = 1 for sparse_full_model) |
| `mask` | `(T,)` | Binary mask: 1 = valid token, 0 = padding |
| `positions` | `(T, 3)` | 3-D structural address for each token |

`T` is the total number of tokenised parameters. For a CNN with many layers and channels this can be in the hundreds of thousands.

The tokeniser used in most configs is `sparse_full_model`, which includes every parameter in the state dict as a single-value token.

---

## 6. Windowing

**File:** [src/sane/data/datasets/windowed_dataset.py](src/sane/data/datasets/windowed_dataset.py)

Because transformers have a fixed context length, the full token sequence of a checkpoint is split into overlapping or non-overlapping **windows** of length `N` (e.g. 512 or 2048).

### Window structure

```python
Window = Tuple[
    torch.Tensor,   # tokens    (N, token_dim)
    torch.Tensor,   # mask      (N,)
    torch.Tensor,   # positions (N, 3)
]
```

### Windowing modes

| Config key | Meaning |
|------------|---------|
| `num_windows_per_model` | Fixed number of windows sampled per checkpoint |
| `num_windows_per_layer` | Windows sampled per layer (takes precedence when set) |

### Window placement strategies

| Strategy | Behaviour |
|----------|-----------|
| `"consecutive"` | Windows placed sequentially from the start of the token sequence |
| `"distributed"` | Windows spread evenly across the token sequence |

### Key methods

- `__getitem__(index)` — returns one `Window` (used by the training `DataLoader`)
- `__getmodel__(index)` — returns **all windows** for checkpoint at `index` (used by `export_latent.py`)

### Caching

`CachedWindowedDataset` wraps `WindowedDataset` with an optional disk cache to avoid re-tokenising on every epoch:

| `cache_mode` | Behaviour |
|--------------|-----------|
| `"none"` | No caching, generate windows in memory each time |
| `"per_model"` | One `.pt` file per checkpoint |
| `"per_window"` | One `.pt` file per window (maximises I/O reduction) |

The cache is keyed by a hash of the full data config, so it is automatically invalidated when any config parameter changes.

---

## 7. Loss Functions

**File:** [src/sane/loss/](src/sane/loss/)

All losses inherit from `SANELoss(nn.Module, ABC)` and return a `Dict[str, torch.Tensor]` that always contains the key `"loss"` (the term to backprop) plus metric scalars.

### 7.1 MSEReconstructionLoss

**File:** [src/sane/loss/reconstruction.py](src/sane/loss/reconstruction.py)

Penalises the L2 distance between the input window `x` and the reconstructed window `x̂`.

Optional normalization modes:

| `normalize` | Effect |
|-------------|--------|
| `"token"` | Normalise MSE per token position |
| `"window"` | Normalise by total window size |
| `"batch"` | Normalise by batch statistics |
| `None` | Raw MSE |

Returned dict keys:

| Key | Content |
|-----|---------|
| `"loss"` | Main scalar loss |
| `"loss_mse_reconstruction"` | Same as above |
| `"loss_mse_reconstruction_rsq"` | R² score (measures reconstruction quality) |
| `"mask_ratio"` | Fraction of valid (non-padded) tokens |
| `"compression_ratio"` | `latent_size / input_size` |

### 7.2 NTXentLoss (Contrastive)

**File:** [src/sane/loss/contrastive.py](src/sane/loss/contrastive.py)

Normalised Temperature-scaled Cross Entropy (NT-Xent / SimCLR-style). Maximises cosine similarity between two projected embeddings of the **same** checkpoint and minimises it across different checkpoints in the batch.

- Inputs: `z_p_i`, `z_p_j` — each `(B, D)`
- Returns: `{"loss": nt_xent, "loss_contrastive": nt_xent}`

### 7.3 CompositeLoss

**File:** [src/sane/loss/loss.py](src/sane/loss/loss.py)

Linearly combines multiple `SANELoss` instances with configurable weights. Supports an automatic "residual" weight that makes all weights sum to 1.

```python
total_loss = alpha * loss_reconstruction + beta * loss_contrastive
```

---

## 8. Training Setup

**Files:** [src/sane/trainer/trainer.py](src/sane/trainer/trainer.py), [src/sane/trainer/contrastive_trainer.py](src/sane/trainer/contrastive_trainer.py)

### 8.1 SANETrainer

Top-level trainer class. Orchestrates model, data, optimiser, scheduler, and loss.

**Key methods:**

| Method | Purpose |
|--------|---------|
| `setup(config)` | Initialise everything from Hydra config |
| `step()` | Run one full epoch (train + optional validation) |
| `_step_training()` | Iterate all training batches |
| `_step_training_forward()` | Single-batch forward + backward + optimiser step |
| `_step_validation()` | Evaluate on validation set |

### 8.2 Single-Batch Forward

```python
# Batch shapes (standard, no augmentations)
tokens  (B, N, token_dim)
mask    (B, N)
pos     (B, N, 3)

# With contrastive augmentations (two views)
tokens  (B, 2, N, token_dim)   → extract view i and view j
pos     (B, 2, N, 3)

# Forward
output = model(tokens, pos, mask)
# output['z']     (B, N, latent_dim)
# output['x_hat'] (B, N, token_dim)
# output['z_p']   (B, latent_dim)  [if projector]

# Loss
loss_dict = loss_fn({'x_i': ..., 'x_hat_i': ..., 'z_p_i': ..., ...})

# Backward
loss_dict['loss'].backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
optimizer.step()
scheduler.step()
```

### 8.3 Optimiser

AdamW with decoupled weight decay. Matrices (`dim >= 2`) get weight decay; biases and layer-norm parameters do not.

```python
decay_params   = [p for p in model.parameters() if p.dim() >= 2]
nodecay_params = [p for p in model.parameters() if p.dim() < 2]
```

### 8.4 Contrastive Training Flow

When augmentations are enabled each batch contains two views of the same checkpoint (produced by weight-matching permutations):

```
                    ┌──────────────────────┐
Checkpoint          │  weight-matching     │
state_dict  ──────► │  permutation × 2     │
                    └──────┬─────────┬─────┘
                           │         │
                      view_i       view_j
                     (B,N,D)      (B,N,D)
                        │            │
                   ┌────▼─────┐  ┌───▼──────┐
                   │ Encoder  │  │ Encoder  │
                   └────┬─────┘  └───┬──────┘
                        │ z_i        │ z_j
                   ┌────▼─────┐  ┌───▼──────┐
                   │Projector │  │Projector │
                   └────┬─────┘  └───┬──────┘
                        │ z_p_i      │ z_p_j
                        └─────┬──────┘
                              │
                          NTXentLoss
                               +
                   MSE(x_i, x̂_i) + MSE(x_j, x̂_j)
```

### 8.5 Key Hyperparameters

| Parameter | Typical value | Effect |
|-----------|---------------|--------|
| `epochs` | 100–500 | Total training epochs |
| `validation_frequency` | 5 | Run validation every N epochs |
| `batch_size` | 128 | Windows per training step |
| `learning_rate` | 1e-4 | Peak LR for OneCycleLR |
| `weight_decay` | 1e-2 | L2 regularisation on weight matrices |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `use_amp` | True | Automatic Mixed Precision |
| `d_model` | 1024 | Internal transformer dimension |
| `num_layers` | 16 | Transformer depth |
| `nhead` | 16 | Attention heads |
| `latent_dim` | 128 | Per-token latent dimension |
| `window_size` | 512–2048 | Tokens per window |

---

## 9. Configuration System

SANE uses [Hydra](https://hydra.cc/) for hierarchical configuration. Configs are stored under [config/](config/) and composed at runtime.

### Example composition

```yaml
# config/experiment.yaml
defaults:
  - model: sane_medium          # model architecture
  - data: cifar10_cnn           # dataset + tokeniser
  - loss: contrastive_recon     # combined loss
  - optimizer: adamw            # optimiser
  - scheduler: onecycle         # LR schedule
```

### Important config groups

| Group | Path | Controls |
|-------|------|----------|
| Model | `config/model/` | Architecture sizes, embedding type |
| Encoder | `config/model/encoder/` | d_model, layers, heads, transformer type |
| Embedder | `config/model/embedder/` | average vs MLP |
| Projector | `config/model/projector/` | Contrastive projection head |
| Data | `config/data/` | Zoo root, splits, batch size |
| Checkpoints dataset | `config/data/checkpoints_dataset/` | epoch_idx, filters, augmentations |
| Tokeniser | `config/data/tokenizer/` | Window size, token dim |
| Tokens dataset | `config/data/tokens_dataset/` | Cache mode and directory |
| Loss | `config/loss/` | Loss type(s) and weights |

---

## 10. export_latent.py

**File:** [scripts/export_latent.py](scripts/export_latent.py)

### Purpose

Takes a **trained SANE checkpoint** and encodes every checkpoint in the model zoo into fixed-size latent vectors. The output is a `.pt` file (one per split) containing the latent representations alongside metadata, ready for downstream use (regression, classification, retrieval, etc.).

### CLI

```bash
python scripts/export_latent.py <sane_checkpoint_path> [output_dir] \
  --mode {average|windows|token}   # embedding granularity (default: average)
  --epochs N [N ...]               # override which zoo epochs to use
  --all-epochs                     # use all available epochs
  --max-checkpoints N              # limit to N checkpoints
  --cache-dir PATH                 # override token cache directory
  --num-windows N|auto             # override number of windows per checkpoint
  --window-strategy {consecutive|distributed}
  --cache-mode {none|per_model|per_window}
  --preload-cache                  # eagerly materialise the cache
  --encode-batch-size N            # windows per GPU forward pass (default: 64)
```

### Encoding Modes

| Mode | Granularity | Output per checkpoint | Use case |
|------|-------------|----------------------|----------|
| `average` | 1 vector | `(latent_dim,)` numpy array | Standard: one embedding per checkpoint |
| `windows` | per window | `Tensor(num_windows, latent_dim)` | Retain window-level detail |
| `token` | per token | `Tensor(num_windows * N, latent_dim)` | Full encoder output, no aggregation |

### What the `average` output represents

In **average mode** (the default), each row in the output embedding matrix is a **single fixed-size vector that encodes the entire state of a trained neural network at a given training epoch**. Concretely:

1. All parameter windows of the checkpoint are encoded individually by the transformer encoder and averaged by the embedder head into `(latent_dim,)` vectors.
2. Those per-window vectors are then **mean-pooled** across all windows of the checkpoint.
3. The result is one `(latent_dim,)` = one `(128,)` vector per checkpoint.

This vector lives in a **weight-space latent manifold** where:
- Checkpoints of the same architecture trained on the same task but with different random seeds cluster together.
- Checkpoints at different training epochs of the same model trace a trajectory through the space.
- The position in this space correlates with generalisation, training dynamics, and task-relevant properties (depending on what was supervised during SANE training).

### Internal processing flow

```python
for ckpt_idx in range(n_checkpoints):

    # 1. Retrieve all tokenised windows for this checkpoint
    windows = dataset.__getmodel__(ckpt_idx)   # List[Window]

    # 2. Build mini-batches for GPU efficiency
    for batch in batches_of(windows, size=encode_batch_size):
        batch_tokens  (B, N, token_dim)   → GPU
        batch_pos     (B, N, 3)           → GPU

        # 3a. average / windows mode: encoder + embedder
        z_emb = model.forward_embeddings(batch_tokens, batch_pos)  # (B, latent_dim)

        # 3b. token mode: raw encoder output
        z = model.encoder(batch_tokens, batch_pos)   # (B, N, latent_dim)
        z_flat = z.flatten(0, 1)                     # (B*N, latent_dim)

    # 4. Aggregate across windows
    if mode == "average":
        all_window_embs.mean(dim=0)         # (latent_dim,)
    elif mode == "windows":
        all_window_embs                     # (num_windows, latent_dim)
    elif mode == "token":
        all_token_embs                      # (num_windows * N, latent_dim)
```

### Output file format

```python
torch.save({
    "mode":             "average" | "windows" | "token",
    "model_ids":        List[str],          # unique model identifiers
    "checkpoint_paths": List[str],          # absolute paths to .pt checkpoint files
    "epoch":            List[int],          # zoo epoch index for each checkpoint
    "embeddings":       np.ndarray          # average: (num_checkpoints, latent_dim)
                        | List[Tensor]      # windows / token: one tensor per ckpt
}, output_path)
```

**File naming convention:**

| Mode | File |
|------|------|
| `average` | `{dataset_name}_latent_representations_{split}.pt` |
| `windows` | `{dataset_name}_latent_representations_{split}_windows.pt` |
| `token` | `{dataset_name}_latent_representations_{split}_tokens.pt` |

All files are written into a subdirectory `{dataset_name}_encoded_checkpoints/` inside the specified output directory.

### Chunking for large datasets

For `windows` and `token` modes on large zoos, the script can flush intermediate results to disk every `chunk_size` checkpoints. The manifest `.pt` file is then a list of chunk-file paths rather than in-memory tensors, allowing the caller to load chunks on demand.

### Position validation

Before each GPU batch the script validates that no position index exceeds the learned embedding table size (derived from `max_positions` in the model config). If a violation is found, a detailed error is raised that shows:
- Which split, checkpoint, epoch, and window triggered the error
- The exact batch indices with out-of-bounds positions
- A comparison table of actual min/max vs allowed limits per dimension

This prevents silent embedding-table out-of-range lookups that would produce incorrect (but non-crashing) results.

---

## 11. Full Data Flow: Raw Checkpoint → Latent Embedding

```
┌──────────────────────────────────────────────────────────────────┐
│  DISK:  /zoo_root/model_42/epoch_30/checkpoints/state_dict.pt    │
└──────────────────────────────┬───────────────────────────────────┘
                                │
                        ZooDataset.load()
                                │
                  ┌─────────────▼─────────────────┐
                  │  state_dict: OrderedDict      │
                  │  {layer.weight: Tensor, ...}  │
                  └─────────────┬─────────────────┘
                                │
                        tokeniser.tokenize()
                                │
         ┌──────────────────────▼──────────────────────────┐
         │  tokens    (T, 1)   — raw float weight values   │
         │  mask      (T,)     — valid / padding flag      │
         │  positions (T, 3)   — (layer, submodule, param) │
         │  T ≈ 100K–10M depending on architecture         │
         └──────────────────────┬──────────────────────────┘
                                │
                      WindowedDataset.window()
                                │
         ┌──────────────────────▼───────────────────────────┐
         │  Window_0:  tokens (N,1), mask (N,), pos (N,3)   │
         │  Window_1:  ...                                  │
         │  ...                                             │
         │  N = window_size (e.g. 1024)                     │
         └──────────────────────┬───────────────────────────┘
                                │
                    DataLoader.collate() / __getmodel__()
                                │
         ┌──────────────────────▼──────────────────────────┐
         │  batch_tokens  (B, N, 1)                        │
         │  batch_mask    (B, N)                           │
         │  batch_pos     (B, N, 3)                        │
         └──────────────────────┬──────────────────────────┘
                                │
             SANETransformerEncoder.forward(tokens, pos)
                                │
         ┌──────────────────────▼────────────────────────────┐
         │  input_layer:          (B, N, 1) → (B, N, 1024)   │
         │  + position_embedding: fuse (B, N, 3) positions   │
         │  transformer (16L×16H): context across N tokens   │
         │  output_layer:         (B, N, 1024) → (B, N, 128) │
         └──────────────────────┬────────────────────────────┘
                                │  z  (B, N, 128)
                                │
             AverageProjectionHead.forward(z)
                                │  mean over N
         ┌──────────────────────▼────────────────────────────┐
         │  z_emb  (B, 128)  — one vector per window         │
         └──────────────────────┬────────────────────────────┘
                                │
                   aggregate over all windows
                                │
         ┌──────────────────────▼────────────────────────────┐
         │  AVERAGE MODE:  mean → (128,)   per checkpoint    │
         │  WINDOWS MODE:  stack → (W, 128) per checkpoint   │
         │  TOKEN MODE:    raw encoder z, (W×N, 128)         │
         └──────────────────────┬────────────────────────────┘
                                │
                 save to .pt file  (see Section 10)
```

---

## 12. Shapes and Dimensions Reference

| Tensor | Shape | Description |
|--------|-------|-------------|
| Raw tokens | `(T, 1)` | Full tokenised checkpoint; T can be millions |
| Window tokens | `(N, 1)` | Single window; N = window_size |
| Batch tokens | `(B, N, 1)` | Training batch |
| Batch with views | `(B, 2, N, 1)` | Two augmented views for contrastive training |
| Batch positions | `(B, N, 3)` | Layer / submodule / param-index per token |
| After input_layer | `(B, N, 1024)` | Projected to transformer d_model |
| After transformer | `(B, N, 1024)` | Contextualised representations |
| Encoder output z | `(B, N, 128)` | Per-token latent; latent_dim = 128 |
| After embedder | `(B, 128)` | Per-window embedding |
| Exported average | `(C, 128)` | C = total checkpoints in split |
| Exported windows | `List[(W_i, 128)]` | W_i windows for checkpoint i |
| Exported tokens | `List[(W_i×N, 128)]` | All encoder token outputs |
| Pos. embed. table | `(max_pos[d], 512)` | Per-dimension; default max_pos = [1024, 256, 256] |
