# Architecture Reference

Reference for the config system, models, data pipeline, and training loop.

---

## Config System

All configs use `ml_collections.ConfigDict`.

**Pattern**: experiment config calls `base_config()` from `configs/base.py`, overrides fields, adds `model.*` params, calls `.lock()`. Must export `get_config()`.

```python
# configs/my_experiment.py
from configs.base import base_config

def get_config():
    cfg = base_config()
    cfg.model.arch = "stft_cnn_bilstm"
    cfg.model.checkpoint_name = "My_Experiment_v1"
    cfg.model.hidden_dim = 64
    # ...
    cfg.lock()
    return cfg
```

`train.py` loads configs dynamically: `importlib.import_module(f"configs.{args.config}")`.

### Key Config Groups

| Group | Key fields |
|---|---|
| `eeg` | `in_channels` (22), `num_classes` (5) |
| `preprocessing` | `representation` ("stft"/"raw"), `resample_rate` (250), `freq_fork` (4,40), `t_fork` (0,6), `window_sec` (3.0), `window_overlap` (0.9), `stft_nperseg` (64), `stft_overlap` (48), `use_cache`, `cache_path` |
| `regularization` | `clip_sigma` (4.0), `gaussian_std` (0.15), `scale_fork` (0.8,1.2), `max_shift` (10), `channel_dropout` (0.2) |
| `data` | `dataset` ("our5class"/"bnci2014001"), `data_path`, `num_workers` (8) |
| `model` | `arch` (selects from registry), `checkpoint_name`, + arch-specific params |
| `training` | `optimizer`, `loss_function`, `label_smoothing` (0.2), `batch_size` (64), `epochs` (1000), `lr` (3e-4), `weight_decay` (0.1), `gradient_clip_norm` (2.0), `es_patience` (48), `rollback_patience` (16), `lr_patience` (16), `lr_factor` (0.5), `min_lr` (1e-7) |

---

## Models

All models implement `from_config(cls, cfg)` classmethod. Registered in `_MODEL_REGISTRY` in `src/models/__init__.py`. `build_model(cfg, device)` returns `(model, criterion, optimizer, scheduler)`.

### STFT_CNN_BiLSTM (`src/models/CRNN.py`)

Input: `(B, 22, 33, T_stft)` - channels as "input channels" to Conv2d.

```
Conv2d(22, hidden, kernel=(freq_bins, 1))   # collapse entire freq dim
-> BN -> ELU -> Dropout
-> SEBlock2D(hidden, reduction)             # channel attention
-> squeeze freq dim, permute to (B, T, hidden)
-> BiLSTM(hidden, hidden, layers=2, dropout)
-> mean over time -> (B, 2*hidden)
-> Linear(2*hidden, num_classes)
```

Config params: `hidden_dim=64`, `rnn_layers=2`, `dropout=0.5`, `se_reduction=4`.

### Raw_CNN_BiLSTM (`src/models/RawCRNN.py`)

Input: `(B, 22, 750)` - 3s at 250 Hz.

```
Conv1d(22, hidden, k=25, pad=12)   # ~100ms receptive field
-> BN -> ELU -> Dropout
-> Conv1d(hidden, hidden, k=13, pad=6)
-> BN -> ELU -> Dropout
-> SE1DBlock(hidden, reduction)
-> AvgPool1d(pool_factor)          # 750 -> ~94 steps
-> permute to (B, T//pool, hidden)
-> BiLSTM(hidden, hidden, layers=2, dropout)
-> mean over time -> (B, 2*hidden)
-> Linear(2*hidden, num_classes)
```

Config params: same as STFT variant + `pool_factor=8`.

### STFT_ResNet18 (`src/models/ResNet18.py`)

Input: `(B, 22, H, W)` - any STFT shape.

```
Stem: Conv2d(22, 64, k=7, s=2, p=3) -> BN -> SiLU -> MaxPool
4 layers: [64->64, 64->128, 128->256, 256->512] x 2 ResidualBlocks each
  ResidualBlock: Conv->BN->SiLU->Conv->BN->SEBlock(reduction=16)->add->SiLU
AdaptiveAvgPool2d(1,1) -> Dropout -> Linear(512, num_classes)
```

Config params: `dropout=0.5`. SE reduction hardcoded to 16. Kaiming init. ~11.3M params.

---

## Data Pipeline Details

### MOABB Integration (`src/data_proc/signal_ops.py`)

The sliding window and STFT are injected into MOABB's internal sklearn pipeline at the `ARRAY` stage, before caching. This means cached arrays already contain fully preprocessed windows - no recomputation on subsequent runs.

`inject_array_transforms()` locates MOABB's `ForkPipelines` and appends `FunctionTransformer` steps to both the X and events branches. Labels are expanded by repeating each label `windows_per_trial` times.

### Shapes After Preprocessing

| Representation | Shape | Breakdown |
|---|---|---|
| STFT | `(N, 22, 33, 48)` | N windows, 22 channels, 33 freq bins, 48 STFT time steps |
| Raw | `(N, 22, 750)` | N windows, 22 channels, 750 samples (3s x 250Hz) |

`freq_bins = stft_nperseg // 2 + 1 = 33`. Windows per trial with default params: 21 (3s window, 90% overlap, 6s trial at 250Hz).

### Our5Class Dataset (`src/data_proc/our_5_class.py`)

MOABB-compatible wrapper. Per-run preprocessing: set montage (standard_1020), pick 22 EEG + M1/M2, re-reference to M1/M2 average, drop references, add annotations from CSV markers.

Train/val split: controlled by `get_last_trials_xor` flag. `False` = all runs except last (train), `True` = last run only (val). Factory creates two separate dataset instances.

### Transforms (`src/loaders/transforms.py`)

Applied at DataLoader time via `TransformWrapper`, not at dataset init.

| Transform | What it does | Dim behavior |
|---|---|---|
| `ClipOutliers(sigma)` | Clamp to mean +/- sigma*std | Per last-dim slice |
| `LogCompress` | `log1p(x)` | Elementwise, STFT only |
| `ZScoreNormalize` | `(x - mean) / std` | Global (single scalar mean/std per sample) |

### Augmentations (`src/loaders/augmentation.py`)

Train only. Applied after transforms.

| Augmentation | What it does |
|---|---|
| `GaussianNoise(std)` | Additive N(0, std) noise |
| `RandomScale(lo, hi)` | Per-channel uniform scale factor |
| `TimeShift(max_shift)` | Circular roll on last axis, random offset |
| `ChannelDropout(p)` | Zero out each channel independently with prob p |

---

## Training Loop (`src/train_utils/classic_trainer.py`)

### Per Epoch

1. Forward pass -> CrossEntropyLoss (with label smoothing) -> backward
2. `clip_grad_norm_(max_norm)` -> optimizer step
3. Log train loss/acc to TensorBoard (`runs/{checkpoint_name}`) and in-memory history
4. Validation pass (if val set provided)
5. `EarlyStopping` checks val_loss (patience=48)
6. `ReduceLROnPlateau` steps on val_loss (factor=0.5, patience=16, min_lr=1e-7)
7. `CheckpointManager` rollback check (patience=16)

### Rollback Mechanism

`CheckpointManager` saves best model+optimizer state in memory (or on disk). When val_loss fails to improve for `rollback_patience` epochs, it restores the best state. Key detail: current LR is preserved across rollback (not rewound), since the scheduler may have reduced it.

After rollback, history is also rewound, so `true_epoch` (used for TensorBoard x-axis) resets accordingly.

### Output

Best model saved to `checkpoints/{checkpoint_name}/best.pt` (state dict) + `best.yaml` (config). Training history saved as `best.csv`.

### EarlyStopping

Fires when `counter > patience` (not `>=`), so actual tolerance is `patience + 1` epochs without improvement.
