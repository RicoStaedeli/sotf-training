# scripts/

## export_latent.py

Encodes model zoo checkpoints from all dataset splits using a trained `SANEAutoEncoder` and saves the resulting latent vectors to disk.

### What it does

For each split (`train`, `val`, `test`), the script:

1. Loads the trainer config and model weights from a SANE checkpoint.
2. Reconstructs the full dataset splits via `SANEContrastiveTrainer`.
3. Passes every checkpoint through the encoder + embedder (`sane_ae.forward_embeddings`) to produce one latent vector per sliding window.
4. Averages window-level vectors into a single fixed-size embedding per checkpoint.
5. Saves the result as a `.pt` file containing `model_ids` (list of strings) and `embeddings` (tensor of shape `[N, latent_dim]`).

### Output structure

```
<output_dir>/
└── <dataset_name>_encoded_checkpoints/
    ├── <dataset_name>_latent_representations_train.pt
    ├── <dataset_name>_latent_representations_val.pt
    └── <dataset_name>_latent_representations_test.pt
```

Each `.pt` file can be loaded with:

```python
import torch
data = torch.load("<dataset_name>_encoded_checkpoints/<dataset_name>_latent_representations_train.pt")
embeddings = data["embeddings"]  # shape: [N, latent_dim]
model_ids  = data["model_ids"]   # list of N strings
```

### How to run

```bash
python scripts/export_latent.py <sane_checkpoint_path> [output_dir]
```

| Argument | Description |
|---|---|
| `sane_checkpoint_path` | Path to the `.pt` checkpoint file saved during SANE training (e.g. `experiments/run1/checkpoints/best.pt`). |
| `output_dir` *(optional)* | Root directory where `encoded_checkpoints/` will be created. Defaults to the experiment root inferred from the checkpoint path. |
| `--cache-dir PATH` *(optional)* | Overrides `data.tokens_dataset.cache_dir` from the checkpoint config. Use this when the saved path (for example `/local/cache/tokens`) is not writable on the machine running evaluation. |
| `--epochs N ...` *(optional)* | Overrides the checkpoint epoch indices to encode. Mutually exclusive with `--all-epochs`. |
| `--all-epochs` *(optional)* | Auto-detects and encodes every available epoch in the zoo directory. |
| `--max-checkpoints N` *(optional)* | Overrides `max_checkpoints`; use `null`, `none`, or `0` for no limit. |

**Example:**

```bash
python scripts/export_latent.py experiments/run1/checkpoints/best.pt results/
```

Splits that are not configured (i.e. `None` in the trainer) are skipped automatically.
