# Architecture reference

Config system, models, data pipeline, training loop. Companion to
`data.md` (datasets and preprocessing) and `sweeps.md` (Optuna).

## Config system

All configs use `ml_collections.ConfigDict`. An experiment config calls
`base_config()` from `configs/base.py`, overrides what it needs, adds
`model.*` params, then `.lock()`s. It must export `get_config()`.

```python
# configs/my_experiment.py
from .base import base_config


def get_config():
    cfg = base_config()
    with cfg.unlocked():
        cfg.model.arch = "eegencoder"
        cfg.model.checkpoint_name = "my_experiment"
        cfg.model.f1 = 16
        # ...
    cfg.lock()
    return cfg
```

`train.py`, `train_subjects.py`, and `sweep.py` load configs
dynamically: `importlib.import_module(f"configs.{args.config}")`.

### Config groups

| Group | Key fields |
|---|---|
| `eeg` | `in_channels` (22), `num_classes` (5 for Our5, 4 for BCI 2a) |
| `preprocessing` | `representation` (`raw` / `stft`), `resample_rate` (250), `freq_fork` (4, 40), `t_fork`, `window_sec` (2.0), `window_overlap` (0.95), `stft_nperseg` (64), `stft_overlap` (48), `normalize` (`per_subject` / `per_channel` / `per_window`), `use_cache`, `cache_path` |
| `regularization` | `clip_sigma` (5.0), `gaussian_std` (0.0), `scale_fork` (1.0, 1.0), `max_shift` (0), `channel_dropout` (0.0), `mixup_alpha` (0.0) |
| `data` | `dataset` (`our5class` / `our4class` / `bnci2014001`), `data_path`, `split_mode`, `subjects` (optional), `num_workers` |
| `model` | `arch` (registry key), `checkpoint_name`, + arch-specific params |
| `training` | `optimizer`, `loss_function`, `label_smoothing` (0.1), `batch_size` (64), `epochs` (200), `num_runs`, `lr` (1e-3), `weight_decay` (0.0), `gradient_clip_norm` (2.0), `es_patience` (32), `rollback_patience` (16), `lr_patience` (16), `lr_factor` (0.5), `min_lr` (1e-7) |

Defaults are set by `base_config()`. The six winner configs re-state
every field they depend on, so the headline kappa does not silently
shift if `base_config()` changes.

`cfg.preprocessing.t_fork` is measured from MOABB's natural anchor:
trial start for Our5Class (`interval[0] = 0`) and cue onset for
BNCI2014_001 (`interval[0] = 2`). The cue-relative `t_fork` convention
is documented in `src/loaders/factory.py:_paradigm`.

## Models

All models implement `from_config(cls, cfg)`. Registered in
`_MODEL_REGISTRY` in `src/models/__init__.py`. `build_model(cfg, device)`
returns `(model, criterion, optimizer, scheduler)`.

| Key | File | Notes |
|---|---|---|
| `eegencoder` | `EEGEncoder.py` | n=5 parallel DSTS branches, ~175 k params |
| `eegnet` | `EEGNet.py` | EEGNet-8,2; ~3 k params |
| `raw_cnn_bilstm` | `RawCRNN.py` | Conv1d + SE + temporal pool + BiLSTM, ~257 k params |
| `raw_resnet18` | `RawResNet18.py` | 1D ResNet-18 + SE |
| `raw_conformer` | `RawConformer.py` | EEG-Conformer (conv tokens + transformer) |
| `stft_cnn_bilstm` | `CRNN.py` | STFT analogue of the CNN-BiLSTM, ~215 k params |
| `stft_resnet18` | `ResNet18.py` | 2D ResNet-18 + SE on spectrograms, ~11.3 M params |

### EEGEncoder (`src/models/EEGEncoder.py`, `_blocks.py`)

Input `(B, C, T)` raw.

```
DownsamplingProjector (3-stage Conv2d, EEGNet-style depthwise / separable
    with two AvgPools)
    -> n=5 parallel DSTS branches, each:
        Dropout -> CausalTCN(last timestep) +
                   StableTransformer(time-mean, 2 layers, 2 heads,
                                     RMSNorm + SwiGLU + rotary + causal mask)
        -> sum, per-branch Linear -> num_classes
    -> average branch logits
```

`_blocks.py` carries the reusable kernel (`RmsNorm`, `SwigluFfn`,
`RotaryEmbedding`, `CausalSelfAttention`, `StableTransformerBlock`,
`CausalTcn`).

### EEGNet (`src/models/EEGNet.py`)

Canonical EEGNet-8,2 (Lawhern et al.). Optional `endpool_ms` replaces
the terminal flatten with an average over the most-recent `endpool_ms`
of input - used by the BCI 2a `w=1 s` cell.

### Raw_CNN_BiLSTM (`src/models/RawCRNN.py`)

```
Conv1d(C, hidden, k=25, pad=12)         # ~100 ms RF
    -> BN -> ELU -> Dropout
    -> Conv1d(hidden, hidden, k=13, pad=6)
    -> BN -> ELU -> Dropout
    -> SE1DBlock(hidden, reduction)
    -> AvgPool1d(pool_factor)
    -> permute -> BiLSTM(hidden, layers, dropout)
    -> mean over time (or last step if causal) -> Linear -> num_classes
```

Config: `hidden_dim`, `rnn_layers`, `dropout`, `pool_factor`,
`se_reduction`, `causal`.

### Other models

`Raw_ResNet18` is the 1D analogue of the STFT ResNet-18.
`Raw_Conformer` is the convolutional + self-attention encoder
(Song et al., 2023). `STFT_CNN_BiLSTM` and `STFT_ResNet18` cover the
spectrogram pipeline - no current winner config uses them, but both
stay registered as canonical STFT baselines.

## Data pipeline

```
Raw recordings (Poly5 1000 Hz, 22ch + marker CSVs;
                or MOABB BNCI2014_001)
    -> MOABB MotorImagery paradigm
       (bandpass freq_fork, resample to resample_rate, epoch around t_fork)
    -> Sliding window (window_sec s, window_overlap fraction)
    -> [stft] scipy.signal.stft magnitude (nperseg=64, noverlap=48)
              -> (n_windows, C, 33, 48)
    -> [raw]  windowed signal
              -> (n_windows, C, window_samples)
    -> ClipOutliers -> [LogCompress for stft]
    -> per-subject normalizer (fit on train split,
       applied at sample-fetch via TransformWrapper)
    -> augmentation (only if non-zero knobs; train-only)
    -> DataLoader (persistent_workers, prefetch_factor=4)
```

`src/loaders/factory.py:cache_tag(cfg)` builds a deterministic tag
from every preprocessing key that affects cached arrays (`t_fork`,
`window_sec`, `window_overlap`, `freq_fork`, `split_mode`). Two
configs that differ on any of those never collide.

Per-window shapes for the defaults (`window_sec = 2.0`,
`resample_rate = 250`, `stft_nperseg = 64`):

| Representation | Shape | Breakdown |
|---|---|---|
| stft | `(N, 22, 33, ~13)` | 33 freq bins, T_stft depends on window |
| raw | `(N, 22, 500)` | 22 channels, 500 samples (2 s x 250 Hz) |

## Training loop (`src/train_utils/classic_trainer.py`)

Per epoch:

1. Forward pass -> CrossEntropyLoss (with label smoothing) -> backward.
2. `clip_grad_norm_(max_norm)` -> optimizer step.
3. Log train scalars to TB (`runs/<checkpoint_name>/...`).
4. Validation pass (if a val set was provided).
5. `EarlyStopping(patience)` on val_loss; raises stop when exhausted.
6. `ReduceLROnPlateau(factor, patience, min_lr)` steps on val_loss.
7. `CheckpointManager(rollback_patience)` rolls model weights back to
   the best-val-loss state when patience expires. The current LR is
   preserved (the scheduler may have reduced it in the meantime).

### Sweep mode

`sweep_mode=True` writes nothing to disk except a tempdir cleaned up
after the trial. No TB, no persistent checkpoints, no `add_hparams`.
Used by `sweep.py`.

### Prediction dump

Setting `dump_predictions_path` writes a single `predictions.npz`
after training: per-window val logits, labels, per-window metadata
(`subject_ids`, `windows_per_trial`, `sfreq`, ...). `train_subjects.py`
always sets this; `src/eval/aggregate.py` consumes it.

### Output

- Standard mode - best `model.pt` + config `yaml` in
  `checkpoints/<checkpoint_name>/`; TB scalars under
  `runs/<checkpoint_name>/<dataset>_<ts>/`.
- Per-subject - per-cell layout under `runs/<config>/subject_<N>/seed_<S>/`
  (events + predictions.npz + result.json) plus
  `checkpoints/<config>/subject_<N>/seed_<S>/best.pt`.
- Pooled multi-seed - one TB run per seed under
  `runs/<config>/<dataset>_<ts>_avg<N>/seed_<seed>/`, with a sibling
  `_agg/` carrying mean / std / `n_active` curves and the mean
  confusion matrix.

## Build helpers

- `src/loaders/build_datasets(cfg) -> (train_ds, val_ds, normalizer)`.
- `src/train_utils/build_transforms(cfg, normalizer) -> (train_t, val_t)`.
  Zero-knob augmentations are dropped (not no-op'd), so they consume
  no RNG state.
- `src/models/build_model(cfg, device) -> (model, criterion, optimizer,
  scheduler)`.
- `src/eval/aggregate.aggregate(runs_dir) -> DataFrame`. Walks per-cell
  `result.json` siblings of `predictions.npz` and computes per-window
  and per-action metrics.
