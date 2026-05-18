# Headline kappa

Within-subject motor-imagery results. Each cell trains one model per
`(subject, seed)` under the deployment-honest protocol: last-poly5 split
(Our5Class) or last-run split (BCI 2a), 95 % overlap sliding window,
per-subject scalar z-score, no augmentation. Reported as mean +- std
across subjects of per-seed means. Per-action kappa aggregates the
windows of one trial by mean-of-logits.

## Numbers

| dataset    | arch         | window | schedule      | n_subj x n_seed | win_kappa       | trial_kappa       |
|---|---|---|---|---|---|---|
| Our5Class  | EEGEncoder   | 2.0 s  | 200 ep + ES32 | 14 x 3          | 0.158 +- 0.123  | **0.226 +- 0.172** |
| Our5Class  | EEGNet       | 1.0 s  | 200 ep + ES32 | 14 x 3          | 0.112 +- 0.078  | 0.192 +- 0.142  |
| Our5Class  | RawCNNBiLSTM | 2.0 s  | 200 ep + ES32 | 14 x 3          | 0.105 +- 0.086  | 0.143 +- 0.143  |
| BCI 2a     | EEGEncoder   | 2.0 s  | 200 ep + ES32 | 9 x 3           | 0.548 +- 0.212  | 0.576 +- 0.210  |
| BCI 2a     | EEGEncoder   | 2.0 s  | 500 ep no ES  | 9 x 3           | 0.568 +- 0.226  | 0.604 +- 0.223  |
| BCI 2a     | EEGNet       | 1.0 s  | 200 ep + ES32 | 9 x 3           | 0.457 +- 0.173  | 0.620 +- 0.198  |
| BCI 2a     | EEGNet       | 1.0 s  | 500 ep no ES  | 9 x 3           | 0.467 +- 0.173  | **0.624 +- 0.205** |
| BCI 2a     | RawCNNBiLSTM | 2.0 s  | 200 ep + ES32 | 9 x 3           | 0.457 +- 0.279  | 0.489 +- 0.288  |

Protocol:

- Our5Class - `t_fork = (0, 6)`, 5-class, `last_poly5` split.
- BCI 2a - `t_fork = (1.0, 4.0)` (MI-only after MOABB's cue-anchor
  offset; see `src/loaders/factory.py:_paradigm`), 4-class, `last_run`
  split.

## Reproduce

Each row is `train_subjects.py --config <X> --subjects <S> --seeds <Z>`,
then `python -m src.eval.aggregate --runs-dir runs/<X>`. The `--no-early-stop`
flag is added for the `500 ep no ES` rows.

```bash
conda activate snake

# Our5Class, EEGEncoder, 2 s windows.
python train_subjects.py --config raw_our5_eegencoder__winner_live \
    --subjects 1-14 --seeds 4269-4271
python -m src.eval.aggregate --runs-dir runs/raw_our5_eegencoder__winner_live

# Our5Class, EEGNet, 1 s windows.
python train_subjects.py --config raw_our5_eegnet__winner_live \
    --subjects 1-14 --seeds 4269-4271
python -m src.eval.aggregate --runs-dir runs/raw_our5_eegnet__winner_live

# Our5Class, RawCNNBiLSTM, 2 s windows.
python train_subjects.py --config raw_our5_cnnbilstm__winner_live \
    --subjects 1-14 --seeds 4269-4271
python -m src.eval.aggregate --runs-dir runs/raw_our5_cnnbilstm__winner_live

# BCI 2a, EEGEncoder, 2 s windows, 200 ep + ES.
python train_subjects.py --config raw_bci_eegencoder__winner_live \
    --subjects 1-9 --seeds 4269-4271

# BCI 2a, EEGEncoder, 2 s windows, 500 ep no ES.
python train_subjects.py --config raw_bci_eegencoder__winner_live \
    --subjects 1-9 --seeds 4269-4271 --max-epochs 500 --no-early-stop

# BCI 2a, EEGNet, 1 s windows, 200 ep + ES.
python train_subjects.py --config raw_bci_eegnet__winner_live \
    --subjects 1-9 --seeds 4269-4271

# BCI 2a, EEGNet, 1 s windows, 500 ep no ES (BCI 2a headline cell).
python train_subjects.py --config raw_bci_eegnet__winner_live \
    --subjects 1-9 --seeds 4269-4271 --max-epochs 500 --no-early-stop

# BCI 2a, RawCNNBiLSTM, 2 s windows.
python train_subjects.py --config raw_bci_cnnbilstm__winner_live \
    --subjects 1-9 --seeds 4269-4271
```

`train_subjects.py` is idempotent on `result.json` - re-running the
same command skips finished cells. The aggregator writes `summary.csv`,
`summary_per_subject.csv`, `summary_per_config.csv`, and `summary.md`
into the runs directory.

## Per-subject reference

Per-(subject, config) `trial_kappa` for the Our5Class headline cells
(mean over seeds 4269, 4270, 4271). Subjects 7, 8, 12 are the
consistently hardest cells on Our5Class.

| subject | Our5 EEGEncoder w=2s | Our5 EEGNet w=1s | Our5 RawCNNBiLSTM w=2s |
|---|---|---|---|
| 1  | 0.060   | 0.143  | 0.167 |
| 2  | 0.429   | 0.345  | 0.190 |
| 3  | 0.143   | 0.083  | 0.143 |
| 4  | 0.476   | 0.333  | 0.095 |
| 5  | 0.119   | 0.107  | 0.048 |
| 6  | 0.440   | 0.310  | 0.321 |
| 7  | 0.024   | 0.060  | 0.012 |
| 8  | -0.036  | 0.000  | -0.048|
| 9  | 0.131   | 0.024  | 0.036 |
| 10 | 0.417   | 0.440  | 0.452 |
| 11 | 0.190   | 0.167  | 0.155 |
| 12 | 0.214   | 0.119  | -0.024|
| 13 | 0.155   | 0.190  | 0.131 |
| 14 | 0.398   | 0.370  | 0.324 |
| mean (std) | **0.226 +- 0.172** | 0.192 +- 0.142 | 0.143 +- 0.143 |

Per-subject kappa for the BCI 2a headline cell (EEGNet w=1 s, 500 ep
no ES). Per-window first, per-action mean below:

| subj | win_kappa | win_acc |
|---|---|---|
| 1   | 0.669     | 0.752   |
| 2   | 0.309     | 0.482   |
| 3   | 0.703     | 0.777   |
| 4   | 0.336     | 0.502   |
| 5   | 0.431     | 0.573   |
| 6   | 0.245     | 0.434   |
| 7   | 0.640     | 0.730   |
| 8   | 0.574     | 0.681   |
| 9   | 0.409     | 0.557   |
| mean | 0.480    | 0.611   |
| trial mean | 0.624 | 0.718 |

Per-action mean-of-logits aggregation gives +0.03 to +0.16 kappa
depending on `windows_per_trial`. A 2 s window on a 6 s trial gives
`wpt = 41` for Our5Class and `wpt = 11` for BCI 2a, and lifts kappa
by ~0.06; a 1 s window on BCI 2a gives `wpt = 42` and lifts kappa by
up to 0.16.
