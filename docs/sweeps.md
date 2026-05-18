# Sweep framework

Reference for the config-based Optuna sweep framework (`sweep.py` and
`sweeps/`).

## Design

A sweep is a thin description of a search space on top of an existing
base config. The base config defines the dataset, preprocessing, and
all architectural defaults; the sweep file only declares which fields
vary, and over what range.

| File | Role |
|---|---|
| `sweep.py` | CLI entry point. Loads the sweep module, builds datasets once, drives Optuna. |
| `sweeps/<name>_sweep.py` | Per-sweep search space (`BASE_CONFIG`, `METRIC`, `DIRECTION`, `MAX_EPOCHS`, `define_space`). |
| `sweeps/.studies/<name>.db` | SQLite Optuna storage. Created on first run, auto-resumed afterwards. |
| `configs/<base>__best__<study>.py` | Auto-generated winner config emitted at study end. |

### Invariants

- **Datasets are built once.** `build_datasets(base_cfg)` is called
  before the objective and the resulting tensors are reused across all
  trials. Per-trial cost stays dominated by training, not preprocessing.
- **Sweeps cannot override `data.*` or `preprocessing.*`.** `sweep.py`
  validates this on every trial and raises if violated - those fields
  determine cache keys and tensor shapes.
- **No TensorBoard or persistent checkpoints during sweeps.** The
  trainer runs with `sweep_mode=True`; per-trial artifacts go into a
  tempdir that is removed afterwards.
- **Each trial is reseeded.** `cfg.seed = base_seed + trial.number`,
  so trials are reproducible but not identical.

## Running

```bash
conda activate snake

# 100 trials, single GPU, sequential, ASHA pruning.
python sweep.py --sweep raw_cnnbilstm_sweep --trials 100

# 2 trials x 3 epochs plumbing check, written to <study>_smoke.
python sweep.py --sweep raw_cnnbilstm_sweep --smoke

# Resumable named study (the same name continues where it left off).
python sweep.py --sweep raw_cnnbilstm_sweep --trials 200 --study-name lr_v2

# Wall-clock budget (stop accepting new trials after N seconds).
python sweep.py --sweep raw_cnnbilstm_sweep --trials 500 --timeout 7200

# Retrain the winner at full epoch budget.
python train.py --config raw_our5_cnnbilstm__winner_live__best__raw_cnnbilstm_sweep
```

### CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--sweep` | required | Module name under `sweeps/` (with or without `sweeps.` prefix). |
| `--trials` | 100 | Trials to run this invocation. Existing trials in the DB are kept. |
| `--timeout` | none | Wall-clock seconds budget; trials stop being accepted once exceeded. |
| `--study-name` | `--sweep` | Optuna study name and SQLite filename. Different names = independent studies. |
| `--smoke` | off | 2 trials x 3 epochs, study name suffixed with `_smoke`. |
| `--min-resource` | 5 | ASHA `min_resource` (epochs before pruning may begin). |
| `--reduction-factor` | 3 | ASHA `reduction_factor`. |
| `--seed` | base config seed | TPE sampler seed and trial-seed offset. |

## Authoring a sweep

A sweep file is a plain Python module. Required interface:

```python
# sweeps/your_sweep.py

BASE_CONFIG = "raw_our5_cnnbilstm__winner_live"   # importable as configs.<this>
METRIC = "val_acc"                  # key in trainer.history
DIRECTION = "maximize"              # or "minimize"
MAX_EPOCHS = 120                    # per-trial epoch ceiling


def define_space(trial) -> dict:
    """Return {dotted_config_path: suggested_value} for this trial."""
    return {
        "training.lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "training.weight_decay": trial.suggest_float("wd", 1e-4, 0.3, log=True),
        "training.batch_size": trial.suggest_categorical("bs", [64, 128, 256]),
        "model.dropout": trial.suggest_float("dropout", 0.1, 0.6),
        # ...
    }
```

### Rules

- **Keys are dotted paths into the base config.** `sweep.py` walks the
  path and `setattr`s the leaf with `cfg.unlocked()`, so locked configs
  are fine.
- **Names passed to `trial.suggest_*` are independent from the dotted
  paths.** They appear in Optuna logs / DB as the parameter names and
  are used to write the winner config. Use short names (`lr`, `wd`,
  `dropout`) and let the dotted path carry the location.
- **No `data.*` or `preprocessing.*`.** Those would invalidate the
  prebuilt datasets, and the validator fails the trial. If you need
  to sweep them, run a new study with a different base config or
  refactor to pre-build per-bucket datasets.
- **`MAX_EPOCHS` is the per-trial ceiling.** Early stopping and ASHA
  pruning can both terminate trials sooner. Choose it so good trials
  have room to converge but bad trials are cheap.
- **`METRIC` must appear in `trainer.history`.** Currently:
  `val_loss`, `val_acc`, `train_loss`, `train_acc`.

### Pruning

`sweep.py` wires up Optuna's `SuccessiveHalvingPruner` (ASHA) and
reports the metric after every validation epoch via the trainer's
`epoch_callback`:

```python
def epoch_callback(epoch, metrics):
    trial.report(metrics[METRIC], step=epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()
```

Pruning fires during training, so unpromising trials are killed early
without exhausting `MAX_EPOCHS`.

### Sampler

`TPESampler(multivariate=True, group=True, seed=seed)`. Multivariate
plus group lets TPE model correlations across sweep dimensions rather
than treating them independently.

## Winner config

After `study.optimize` returns, `sweep.py` writes
`configs/<base>__best__<study>.py`:

```python
"""Best config from study 'raw_cnnbilstm_sweep'."""

from .raw_our5_cnnbilstm__winner_live import get_config as _base_get_config


def get_config():
    c = _base_get_config()
    with c.unlocked():
        c.training.lr = 0.00042
        c.training.weight_decay = 0.013
        # ...
    c.lock()
    return c
```

The map from Optuna parameter names back to dotted config paths is
recovered by re-running `define_space` against a recording trial in
`_resolve_param_mapping`. If `define_space` does anything fancy
(conditional `suggest_*` calls whose count differs from the returned
dict size), the recovery falls back to an identity mapping and the
emitted config may be incomplete - keep `define_space` straight-line
where possible.

Retrain the winner at full epoch budget with
`python train.py --config <base>__best__<study>`. That run produces
TensorBoard logs and a persistent checkpoint.

## Resuming and parallelism

- **Resumable** - re-invoking `python sweep.py --sweep <s> --study-name <name>`
  opens the same SQLite study (`load_if_exists=True`) and continues.
  New `--trials N` are additional trials on top of what is in the DB.
- **Sequential by design** - trials run with `n_jobs=1` because each
  trial saturates the GPU. Running two `sweep.py` processes against
  the same study DB works (Optuna handles the locking) but is rarely
  useful on a single-GPU box.
- **Cache hygiene** - `cache_tag(cfg)` encodes every preprocessing key
  that affects cached arrays. Sweeps over those keys are blocked by
  the validator; if a future workflow needs to sweep them, pre-build
  per-bucket datasets outside the trial loop.

## Existing sweeps

| Sweep | Base | Search space |
|---|---|---|
| `raw_cnnbilstm_sweep` | `raw_our5_cnnbilstm__winner_live` | lr, wd, ls, bs, gnoise, chdrop, max_shift, dropout, hidden, rnn_layers, pool_factor |
| `raw_eegnet_sweep` | `raw_our5_eegnet__winner_live` | lr, wd, ls, bs, gnoise, chdrop, max_shift, dropout, F1, D, F2, kernel_length, sep_kernel |
