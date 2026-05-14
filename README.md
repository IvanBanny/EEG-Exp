# IvansNet - Motor Imagery EEG Classification

Experimentation framework for motor imagery (MI) classification from EEG.
Supports multiple models, signal representations (STFT spectrograms, raw signal), and datasets
through a unified config-driven pipeline built on MOABB + PyTorch.

**Primary target**: live MI classification from TMSi APEX EEG (22ch, 10-20 montage).

## Quick Start

```bash
pip install -r requirements.txt

# Train with a specific experiment
python train.py --config stft_cnnbilstm
python train.py --config raw_bci_cnnbilstm
python train.py --config raw_bci_eegnet

# Hyperparameter sweep (Optuna + ASHA pruning)
python sweep.py --sweep raw_bci_cnnbilstm_sweep --trials 100
python sweep.py --sweep raw_bci_cnnbilstm_sweep --smoke           # 2 trials x 3 epochs plumbing check
python train.py --config raw_bci_cnnbilstm__best__raw_bci_cnnbilstm_sweep
```

See [docs/sweeps.md](docs/sweeps.md) for the sweep framework reference.

## Monitoring with TensorBoard

Training runs log scalars, hparams, and best metrics to `runs/{checkpoint_name}/{dataset}_{timestamp}/`.

```bash
tensorboard --logdir runs
```

Then open http://localhost:6006. Use the "HParams" tab to compare runs across configs.

## Available Experiments

Experiments come in two flavors: `*` runs on Our5Class, `*_bci_*` runs on BNCI2014001 (BCI IV 2a).

| Config family | Model | Representation | Notes |
|---|---|---|---|
| `stft_cnnbilstm`, `stft_bci_cnnbilstm` | CNN-BiLSTM | STFT | Conv2d collapses freq, SE block, BiLSTM (~215K) |
| `raw_cnnbilstm`, `raw_bci_cnnbilstm` | CNN-BiLSTM | Raw | Two Conv1d + SE + temporal pooling + BiLSTM (~257K) |
| `stft_resnet18`, `stft_bci_resnet18` | ResNet-18 + SE | STFT | Full ResNet-18 with SE in every residual block (~11.3M) |
| `raw_resnet18`, `raw_bci_resnet18` | 1D ResNet-18 + SE | Raw | Conv1d ResNet variant |
| `raw_conformer`, `raw_bci_conformer` | Conformer | Raw | Convolutional + self-attention encoder |
| `raw_eegnet`, `raw_bci_eegnet` | EEGNet | Raw | Compact depthwise-separable baseline |

Configs ending in `__best__<study>` are auto-generated winners from a sweep (see [docs/sweeps.md](docs/sweeps.md)).

## Datasets

- **Our5Class** - 14 subjects, 5 classes (left/right hand, left/right leg, tongue), TMSi APEX Poly5 format. Place raw data in `our_data/our_5_class/subject{1..14}/`.
- **BNCI2014001** - BCI Competition IV 2a, 9 subjects, 4 classes. Auto-downloaded via MOABB.

## Data Pipeline

```
Poly5 (1000 Hz, 22ch) + marker CSVs
  -> MOABB paradigm (bandpass 4-40 Hz, resample 250 Hz, epoch [0, 6]s)
  -> Sliding window (3s, 90% overlap) -> ~52K windows
  -> [STFT] scipy.signal.stft (nperseg=64, noverlap=48) -> (N, 22, 33, 48)
  -> [raw]  windowed signal                              -> (N, 22, 750)
  -> Transforms: ClipOutliers -> LogCompress* -> ZScoreNormalize
  -> Augmentations (train only): GaussianNoise, RandomScale, TimeShift, ChannelDropout
  -> DataLoader (batch=64, 8 workers)
```
*LogCompress applied to STFT only.

All preprocessed data is cached to `moabb_cache/` after the first run.

## Adding a New Experiment

1. Create `configs/your_experiment.py` - call `base_config()`, override fields, `lock()`
2. If new model: implement `from_config(cls, cfg)`, register in `_MODEL_REGISTRY` (`src/models/__init__.py`)
3. If new dataset: add builder in `src/loaders/factory.py`
4. Run `python train.py --config your_experiment`

## Adding a New Sweep

1. Create `sweeps/your_sweep.py` exporting `BASE_CONFIG`, `METRIC`, `DIRECTION`, `MAX_EPOCHS`, and `define_space(trial) -> dict` of dotted-path overrides.
2. Sweeps are forbidden from overriding `data.*` and `preprocessing.*` (datasets are built once outside the trial loop).
3. Run `python sweep.py --sweep your_sweep --trials 100`.
4. After the study, `sweep.py` writes `configs/<base>__best__<study>.py` so you can retrain the winner at full budget via `python train.py --config <base>__best__<study>`.

See [docs/architecture.md](docs/architecture.md) for configs, models, and the training loop, and [docs/sweeps.md](docs/sweeps.md) for the sweep framework.

## Project Structure

```
ivansnet/
├── train.py                    # Training entry point
├── sweep.py                    # Optuna sweep entry point
├── configs/
│   ├── base.py                 # Shared defaults (base_config())
│   ├── {raw,stft}_{model}.py            # Our5Class experiments
│   ├── {raw,stft}_bci_{model}.py        # BNCI2014001 experiments
│   └── *__best__*.py                    # Auto-generated sweep winners
├── sweeps/
│   ├── <name>_sweep.py         # Search-space definitions
│   └── .studies/               # SQLite study files (auto-created)
├── src/
│   ├── data_proc/              # Poly5 reader, MOABB dataset wrapper, signal ops
│   ├── loaders/                # PyTorch datasets, transforms, augmentations, factory
│   ├── models/                 # Model registry + architectures
│   └── train_utils/            # Training loop, early stopping, checkpointing
├── checkpoints/                # Saved .pt weights, .csv history, .yaml configs
├── our_data/our_5_class/       # Raw EEG data (not in repo)
└── moabb_cache/                # Preprocessed array cache (auto-generated)
```

## Key Training Defaults

| Parameter | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, weight_decay=0.1) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=16) |
| Early stopping | patience=48, with rollback (patience=16) |
| Gradient clipping | max_norm=2.0 |
| Label smoothing | 0.2 |
| Max epochs | 1000 (early stopped) |

## Requirements

- Tested with 16 GiB VRAM, 32 GiB RAM

### Troubleshooting: Out of Memory

Reduce memory by adjusting these config values:
```python
config.preprocessing.window_overlap: 0.9 -> 0.8   # fewer windows
config.preprocessing.window_sec: 3.0 -> 4.0       # fewer windows
config.preprocessing.stft_nperseg: 64 -> 128      # fewer STFT time steps
config.preprocessing.use_cache: True -> False     # don't cache to disk
```

## Known Issues

- **Subject 4 label swap** - `our_data/our_5_class/subject4/subject_10_markers.csv` has "Right Hand" and "Left Hand" swapped; fix manually