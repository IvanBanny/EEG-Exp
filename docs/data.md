# Data reference

Datasets, preprocessing pipeline, and cache layout. Companion to
`architecture.md` (config + model + training loop). Loader-side
conventions live inline in the loader modules:
`src/loaders/factory.py:_paradigm` for the BCI 2a anchor offset and
`src/data_proc/our_5_class.py` for the Our5Class preamble.

## Our5Class

In-house TMSi APEX recordings, 14 subjects, 22 EEG channels + M1 / M2
references, 5 classes.

| Property | Value |
|---|---|
| Sampling rate | 1000 Hz raw, 250 Hz after MOABB resample |
| Channels | 22 EEG (10-20 montage) + M1, M2 references |
| Bandpass (default) | 4 - 40 Hz |
| Trial length | ~6 s of motor imagery |
| Classes | `left_hand`, `right_hand`, `left_leg`, `right_leg`, `tongue` |
| Format | TMSi Poly5 binary + companion `_markers.csv` |
| Size | ~4.8 GB raw |

### File layout

```
our_data/our_5_class/
    subject1/
        subject_1.poly5
        subject_1_markers.csv     # time_start_s, time_end_s, body_part
        subject_2.poly5
        subject_2_markers.csv
        ...
    subject2/
    ...
    subject14/
```

### Annotation timing (`PREAMBLE_S = 0.5`)

The recorder writes a fixed 500 ms preamble before the experiment
clock zeroes (`QTimer.singleShot(500, experiment.start)` in
`experimentFINAL.py:362`). The marker CSV stores experiment-clock
timestamps; the loader adds `PREAMBLE_S = 0.5` to each onset so that
experiment-clock t=0 maps to file-clock t=0.5 s, where the MI
actually begins. The raw recording stays uncropped - MOABB epochs
around annotations and ignores the tail.

Sanity warnings fire at load time for files with an abnormally long
tail or a short marker count. The constant lives in
`src/data_proc/our_5_class.py`.

### Train / val split

`split_mode = "last_poly5"` (only supported mode for Our5Class). The
last poly5 file per subject is the validation set; all earlier files
are training. Deployment-honest temporal split - the model never sees
data from the same recording session in both train and val.

### Known data issues

- **Subject 4, `subject_10_markers.csv`** - "Right Hand" and "Left
  Hand" are swapped at the marker source. The loader does not patch
  this. Correct the CSV manually before training a model whose val
  set includes subject 4's last poly5.
- **Subject 13, files 1-3** - paused or restarted recordings, with
  short marker counts and long tails (~50-180 s instead of ~10 s).
  The fix correctly aligns the markers it has, but several trials per
  file are missing. The loader's sanity warnings flag these at load
  time.

## BNCI2014001 (BCI Competition IV 2a)

Standard benchmark, auto-downloaded by MOABB.

| Property | Value |
|---|---|
| Subjects | 9 |
| Channels | 22 EEG + 3 EOG |
| Classes | `left_hand`, `right_hand`, `feet`, `tongue` |
| Paradigm | Motor imagery (sessions `0train`, `1test`) |
| Anchor | Cue onset at trial-time 2 s (`interval = [2, 6]`) |

### `t_fork` convention

MOABB's `MotorImagery(tmin, tmax)` applies
`effective_tmin = self.tmin + dataset.interval[0]` before epoching.
BNCI2014_001 has `interval[0] = 2`, so `t_fork` on this dataset is
cue-relative, not trial-relative. Canonical windows:

| Goal | Trial-time window | `t_fork` |
|---|---|---|
| Reference paper (1.5 s pre-cue + MI) | `[1.5, 6.0]` s | `(-0.5, 4.0)` |
| MI-only (deployment-honest) | `[3.0, 6.0]` s | `(1.0, 4.0)` |

The convention is documented in the `_paradigm` docstring of
`src/loaders/factory.py`.

### Train / val split

| `split_mode` | Train | Val |
|---|---|---|
| `last_run` (default) | all runs except the last per subject | last run per subject |
| `session` | `0train` session | `1test` session |

`last_run` is the deployment-honest within-subject split used by the
headline cells. `session` reproduces the EEGEncoder paper's protocol.

## Preprocessing pipeline

```
1. MOABB MotorImagery paradigm
       freq_fork bandpass, resample to resample_rate Hz, epoch around t_fork.
2. Sliding window
       window_sec, window_overlap fraction.
       Our5Class, t_fork=(0,6), window_sec=2.0, overlap=0.95
           -> wpt = 41 (step = 0.05 s).
       BCI 2a, t_fork=(1,4), window_sec=1.0, overlap=0.95
           -> wpt = 42.
3. Representation
       [raw]  windowed time-domain signal: (N, 22, window_samples).
       [stft] scipy.signal.stft magnitude (nperseg=64, noverlap=48):
              (N, 22, freq_bins, T_stft).
4. Per-window transforms
       ClipOutliers(sigma=clip_sigma) -> [LogCompress for stft].
5. Per-subject normaliser
       Fit on train split, applied to train + val at sample-fetch.
       Modes: per_window | per_subject | per_channel.
6. Augmentation (train only)
       Any of {GaussianNoise, RandomScale, TimeShift, ChannelDropout}
       with non-zero knobs. Zero-knob entries are dropped from the
       pipeline outright.
```

## Cache layout (`moabb_cache/`)

```
moabb_cache/
    <dataset>/
        <representation>/
            <tag>/
                train/  MNE-BIDS-our5-class/...   # Our5Class
                val/    MNE-BIDS-our5-class/...
                MNE-BIDS-bnci2014-001/...         # BCI 2a (split in memory)
```

`<tag>` is `cache_tag(cfg)` from `src/loaders/factory.py`. It folds
every preprocessing key that affects the cached arrays (`t_fork`,
`window_sec`, `window_overlap`, `freq_fork`, `split_mode`). Two configs
that differ on any of those never collide on the same cache. Changing
`representation` is also disjoint (separate `<representation>/`
subtree).

Delete a subtree to force recomputation, or set
`cfg.preprocessing.regen_cache = True` to bust the cache in place.

## Per-window transforms (`src/loaders/transforms.py`)

| Transform | What it does | Applied to |
|---|---|---|
| `ClipOutliers(sigma)` | Clamp to mean +- sigma * std | All representations |
| `LogCompress` | `log1p(x)` | STFT only |
| `ZScoreNormalize` | `(x - mean) / std` per sample | `normalize="per_window"` |
| `PerSubjectZScore` | Pre-fit scalar (mean, std) per subject, applied via `TransformWrapper` | `normalize="per_subject"` |
| `PerChannelZScore` | Pre-fit (mean, std) per (subject, channel) | `normalize="per_channel"` |

## Augmentations (`src/loaders/augmentation.py`, train-only)

| Augmentation | What it does |
|---|---|
| `GaussianNoise(std)` | Additive N(0, std) noise |
| `RandomScale((lo, hi))` | Per-channel uniform scale factor |
| `TimeShift(max_shift)` | Circular roll on last axis, random offset |
| `ChannelDropout(p)` | Zero out each channel independently with prob p |
| `IntraSubjectMixup(alpha)` | Within-(subject, class) Beta-mixup at the dataset level |

The base-config defaults are all-zero (no augmentation). All headline
numbers come from the augmentation-off pipeline.
