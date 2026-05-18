# IvansNet - Motor Imagery EEG Classification

Within-subject motor-imagery classification on TMSi APEX recordings
(Our5Class, 5 classes) and BNCI2014001 (BCI IV 2a, 4 classes). Built
on MOABB + PyTorch with a unified config-driven pipeline.

Headline kappa and reproduction commands live in
[docs/status.md](docs/status.md). Conventions live under [docs/](docs/).

## Quick start

```bash
pip install -r requirements.txt

# Pooled training (one model across subjects, optional multi-seed averaging).
python train.py --config raw_bci_eegencoder__winner_live

# Per-subject training (one model per (subject, seed); the headline regime).
python train_subjects.py --config raw_our5_eegencoder__winner_live \
    --subjects 1-14 --seeds 4269-4271

# Per-action aggregation (writes summary.csv / summary.md / per-subject CSV).
python -m src.eval.aggregate --runs-dir runs/raw_our5_eegencoder__winner_live

# Hyperparameter sweep (Optuna + ASHA pruning, pooled only).
python sweep.py --sweep raw_cnnbilstm_sweep --trials 100
python sweep.py --sweep raw_cnnbilstm_sweep --smoke   # 2 trials x 3 epochs plumbing check
```

TensorBoard scalars land under `runs/<config>/...`:

```bash
tensorboard --logdir runs
```

`docs/training_modes.md` covers when to use pooled vs per-subject;
`docs/sweeps.md` documents the sweep framework end to end.

## Datasets

- **Our5Class** - 14 subjects, 22 ch, 5 MI classes (left / right hand,
  left / right leg, tongue). TMSi APEX Poly5 files + companion marker
  CSVs. Place under `our_data/our_5_class/subject{1..14}/`. The loader
  uses `PREAMBLE_S = 0.5` to align marker timestamps with file-clock
  time; the constant lives in `src/data_proc/our_5_class.py` with a
  short comment on the recorder-side mechanism.
- **BNCI2014001** (BCI IV 2a) - 9 subjects, 22 ch, 4 MI classes.
  Auto-downloaded via MOABB. `t_fork` on this dataset is cue-relative
  (MOABB adds `dataset.interval[0] = 2` to `tmin`); the convention is
  documented in `src/loaders/factory.py:_paradigm`.

## Models (`_MODEL_REGISTRY` in `src/models/__init__.py`)

| Registry key | File | Notes |
|---|---|---|
| `eegencoder` | `EEGEncoder.py` | n=5 parallel DSTS branches (~175 k) |
| `eegnet` | `EEGNet.py` | EEGNet-8,2 baseline (~3 k) |
| `raw_cnn_bilstm` | `RawCRNN.py` | Conv1d + SE + temporal pool + BiLSTM (~257 k) |
| `raw_resnet18` | `RawResNet18.py` | 1D ResNet-18 + SE |
| `raw_conformer` | `RawConformer.py` | EEG-Conformer (conv + self-attention) |
| `stft_cnn_bilstm` | `CRNN.py` | STFT analogue of the CNN-BiLSTM (~215 k) |
| `stft_resnet18` | `ResNet18.py` | 2D ResNet-18 + SE on spectrograms (~11.3 M) |

All models implement `from_config(cls, cfg)`. The six winner configs
(`configs/raw_{bci,our5}_{eegencoder,eegnet,cnnbilstm}__winner_live.py`)
cover the headline cells; new experiments can use any registry key.

## Data pipeline

```
Raw recordings (Poly5 1000 Hz + marker CSVs; or MOABB BNCI2014_001)
    -> MOABB MotorImagery paradigm (bandpass freq_fork Hz, resample 250 Hz,
       epoch around t_fork)
    -> Sliding window (window_sec, window_overlap)
    -> [stft] STFT (nperseg=64, noverlap=48) -> (n_windows, C, 33, 48)
    -> [raw]  windowed signal                -> (n_windows, C, window_samples)
    -> ClipOutliers -> [LogCompress for stft]
    -> per-subject normaliser (fit on train split, applied to train + val)
    -> augmentation (only if non-zero knobs; train-only)
    -> DataLoader (persistent_workers, prefetch_factor=4)
```

The MOABB array cache lives at `moabb_cache/<dataset>/<repr>/<tag>/`,
where `tag` encodes every preprocessing key that affects the cached
arrays (`t_fork`, `window_sec`, `window_overlap`, `freq_fork`,
`split_mode`). Two configs that differ on any of those never collide.

## Adding a new experiment

1. `configs/your_experiment.py` - call `base_config()`, set
   `model.arch`, add model-specific params, `lock()`.
2. New model - add a class with `from_config(cls, cfg)`, register in
   `_MODEL_REGISTRY` (`src/models/__init__.py`).
3. New dataset - add a `_DATASET_EVENTS` entry and a builder branch
   in `src/loaders/factory.py`. Audit the MOABB `interval[0]` anchor
   before setting `t_fork`; the convention is documented in the
   `_paradigm` docstring there.
4. `python train.py --config your_experiment` (pooled) or
   `python train_subjects.py --config your_experiment --subjects ... --seeds ...`.

## Adding a new sweep

1. `sweeps/your_sweep.py` exporting `BASE_CONFIG`, `METRIC`,
   `DIRECTION`, `MAX_EPOCHS`, and `define_space(trial) -> dict` of
   dotted-path overrides.
2. Sweeps cannot override `data.*` or `preprocessing.*` (datasets are
   built once outside the trial loop and reused).
3. `python sweep.py --sweep your_sweep --trials 100`.
4. After the study, `sweep.py` writes
   `configs/<base>__best__<study>.py`. Retrain at full epoch budget
   via `python train.py --config <base>__best__<study>`.

See `docs/sweeps.md` for the full sweep reference.

## Project structure

```
ivansnet/
    train.py                    # Pooled training
    train_subjects.py           # Per-subject training (subject x seed loop)
    sweep.py                    # Optuna sweep
    configs/
        base.py                 # base_config() - shared defaults
        raw_{bci,our5}_{eegencoder,eegnet,cnnbilstm}__winner_live.py
    sweeps/
        <name>_sweep.py
        .studies/               # SQLite study files (auto-created)
    src/
        data_proc/              # Poly5 reader, MOABB Our5Class wrapper, signal ops
        loaders/                # EEGDataset, factory.build_datasets, transforms,
                                #     augmentation
        models/                 # Model registry + architectures
        train_utils/            # ClassicTrainer, build_transforms, early stopping
        eval/                   # aggregate.py - per-action kappa, summaries
    docs/                       # status, training_modes, paper_gap,
                                #     architecture, data, sweeps
    checkpoints/                # Saved .pt weights, .yaml configs
    runs/                       # TensorBoard scalars, predictions, results
    our_data/our_5_class/       # Raw EEG (not in repo)
    moabb_cache/                # Preprocessed array cache (auto-generated)
```

## Key training defaults (`configs/base.py`)

| Parameter | Value |
|---|---|
| Representation | `raw` |
| Window | 2.0 s, 95 % overlap |
| Normaliser | per-subject scalar z-score |
| Augmentation | all off |
| Optimizer | AdamW (lr=1e-3, weight_decay=0.0) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=16) |
| Early stopping | patience 32, rollback patience 16 |
| Gradient clipping | max_norm 2.0 |
| Label smoothing | 0.1 |
| Max epochs | 200 (early-stopped) |

The defaults match the headline winner schedule. Experiment configs
override what they need.

## Requirements

Tested with 16 GiB VRAM, 32 GiB RAM. Sweep trials run sequentially on
a single GPU.

### Troubleshooting: out of memory

```
cfg.preprocessing.window_overlap: 0.95 -> 0.9   # fewer windows
cfg.preprocessing.window_sec: 2.0 -> 3.0        # fewer windows
cfg.preprocessing.stft_nperseg: 64 -> 128       # fewer STFT time steps
cfg.preprocessing.use_cache: True -> False      # do not cache to disk
```

## Known data issues

- **Subject 4 label swap** - `our_data/our_5_class/subject4/subject_10_markers.csv`
  has "Right Hand" and "Left Hand" swapped in the raw markers. The
  loader does not patch this; correct the file manually before
  training a model whose validation includes that subject's last
  poly5 file.
- **Subject 13 files 1-3** - paused or restarted recordings; the
  loader's sanity warnings flag them at load time. Several trials per
  file are missing from the marker CSV. Either drop subject 13
  entirely or drop files 1-3 only.
