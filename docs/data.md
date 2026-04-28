# Data Reference

## Our5Class Dataset

14 subjects recorded with TMSi APEX EEG device. 22 channels (international 10-20 montage) + M1/M2 references.

| Property | Value |
|---|---|
| Sampling rate | 1000 Hz (raw), 250 Hz (after resample) |
| Channels | 22 EEG (Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2, Fpz + M1, M2 ref) |
| Bandpass | 4-40 Hz |
| Trial length | ~6 seconds |
| Classes | left_hand (1), right_hand (2), left_leg (3), right_leg (4), tongue (5) |
| Format | TMSi Poly5 binary + companion CSV markers |
| Size | ~4.8 GB raw |

### File Layout

```
our_data/our_5_class/
  subject1/
    subject_1.poly5          # recording run
    subject_1_markers.csv    # time_start_s, time_end_s, body_part
    subject_2.poly5
    subject_2_markers.csv
    ...                      # ~6 runs per subject
  subject2/
  ...
  subject14/
```

### Known Data Issues

- **Subject 4, file `subject_10_markers.csv`**: "Right Hand" and "Left Hand" labels are swapped. Fix manually before training.

### Train/Val Split

Split by last recording run per subject. All runs except the last form the training set; the last run is validation. This is a temporal split - the model never sees data from the same recording session in both train and val.

## BNCI2014001 (BCI Competition IV 2a)

Standard benchmark dataset. Auto-downloaded by MOABB.

| Property | Value |
|---|---|
| Subjects | 9 |
| Channels | 22 EEG + 3 EOG |
| Classes | left_hand, right_hand, feet, tongue (4 classes) |
| Paradigm | Motor imagery |

Train/val split: by last run per subject (via pandas groupby on run metadata).

To use: set `cfg.data.dataset = "bnci2014001"` and `cfg.eeg.num_classes = 4`. No existing experiment config does this out of the box - you need to create one.

## Preprocessing Flow

### Signal Processing (injected into MOABB pipeline)

The preprocessing is injected into MOABB's sklearn pipeline at the array stage, so all computed windows are cached and don't need recomputation.

1. **MOABB paradigm**: bandpass 4-40 Hz, resample to 250 Hz, epoch [0, 6]s
2. **Sliding window**: 3s window (750 samples), 90% overlap (step = 75 samples), 21 windows per 6s trial
3. **STFT** (if `representation="stft"`): `scipy.signal.stft`, magnitude only (`np.abs(Zxx)`)
   - nperseg=64, noverlap=48 -> 33 freq bins, 48 time steps per window
   - Output: `(n_windows, 22, 33, 48)`
4. **Raw** (if `representation="raw"`): no further transform
   - Output: `(n_windows, 22, 750)`

### Caching

Preprocessed arrays are cached in `moabb_cache/` (configurable via `cfg.preprocessing.cache_path`). Separate cache dirs for train and val splits. Delete the cache dir to force recomputation, or set `regen_cache=True` in `EEGDatasetConfig`.

Cache structure:
```
moabb_cache/
  our5class/
    stft/
      train/    # all runs except last per subject
      val/      # last run per subject
    raw/
      train/
      val/
  bnci2014001/
    full/     # all data, split in memory
```