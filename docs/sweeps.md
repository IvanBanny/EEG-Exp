# Sweep Framework Reference

Reference for the config-based Optuna sweep framework (`sweep.py` + `sweeps/`).

## Design

A sweep is a thin description of a search space on top of an existing base config. The base config defines the dataset, preprocessing, and all architectural defaults; the sweep file only specifies which fields vary and over what range.

| File | Role |
|---|---|
| `sweep.py` | CLI entry point. Loads the sweep module, builds datasets once, drives Optuna. |
| `sweeps/<name>_sweep.py` | Per-sweep search space (`BASE_CONFIG`, `METRIC`, `DIRECTION`, `MAX_EPOCHS`, `define_space`). |
| `sweeps/.studies/<name>.db` | SQLite Optuna storage. Created on first run, auto-resumed afterwards. |
| `configs/<base>__best__<study>.py` | Auto-generated winner config emitted at study end. |

### Key invariants

- **Datasets are built once.** `build_datasets(base_cfg)` is called before the objective and the resulting tensors are reused across all trials in the study. This keeps per-trial cost dominated by training, not preprocessing.
- **Sweeps cannot override `data.*` or `preprocessing.*`.** `sweep.py` validates this at every trial and raises if violated, because those fields determine cache keys and tensor shapes.
- **No TensorBoard / persistent checkpoints during sweeps.** The trainer runs with `sweep_mode=True`: it writes to a tempdir that is cleaned up after each trial.
- **Each trial is reseeded.** `cfg.seed = base_seed + trial.number` so trials are reproducible but not identical.

## Running

```bash
conda activate snake

# Full sweep, 100 trials, single GPU, sequential trials, ASHA pruning
python sweep.py --sweep raw_bci_cnnbilstm_sweep --trials 100

# 2 trials x 3 epochs plumbing check, written to <study>_smoke
python sweep.py --sweep raw_bci_cnnbilstm_sweep --smoke

# Resumable named study (re-running the same name continues where it left off)
python sweep.py --sweep raw_bci_cnnbilstm_sweep --trials 200 --study-name lr_v2

# Wall-clock budget (stop accepting new trials after N seconds)
python sweep.py --sweep raw_bci_cnnbilstm_sweep --trials 500 --timeout 7200

# Retrain the winner at full epoch budget
python train.py --config raw_bci_cnnbilstm__best__raw_bci_cnnbilstm_sweep
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
| `--seed` | base config seed | Used both for the TPE sampler and as the trial-seed offset. |

## Authoring a sweep

A sweep file is a plain Python module. Required interface:

```python
# sweeps/your_sweep.py

BASE_CONFIG = "raw_bci_cnnbilstm"   # importable as configs.<this>
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

- **Keys are dotted paths into the base config.** `sweep.py` walks the path and `setattr`s the leaf with `cfg.unlocked()` so locked configs are fine.
- **Names passed to `trial.suggest_*` are independent from the dotted paths.** They appear in Optuna logs / DB as the parameter names and are also used to write the winner config. Use short names (`lr`, `wd`, `dropout`) and let the dotted path carry the location.
- **No `data.*` or `preprocessing.*`.** Those would invalidate the prebuilt datasets and the run will fail validation. If you need to sweep them, build a new study with a different base config or refactor to pre-build per-bucket datasets.
- **`MAX_EPOCHS` is the per-trial ceiling.** Early stopping and ASHA pruning can both terminate trials earlier. Choose it so that good trials have enough room to converge but bad trials are cheap.
- **`METRIC` must appear in `trainer.history`.** Currently: `val_loss`, `val_acc`, `train_loss`, `train_acc`.

### Pruning behavior

`sweep.py` wires up Optuna's `SuccessiveHalvingPruner` (ASHA) and reports the metric to the trial after every validation epoch via the trainer's `epoch_callback`:

```python
def epoch_callback(epoch, metrics):
    trial.report(metrics[METRIC], step=epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()
```

Pruning fires during training, so unpromising trials are killed early without exhausting `MAX_EPOCHS`.

### Sampler

`TPESampler(multivariate=True, group=True, seed=seed)`. Multivariate + group lets TPE model correlations across the sweep dimensions instead of treating them independently.

## Winner config

After `study.optimize` returns, `sweep.py` writes `configs/<base>__best__<study>.py`:

```python
"""Best config from study 'raw_bci_cnnbilstm_sweep' (base: raw_bci_cnnbilstm)."""

from .raw_bci_cnnbilstm import get_config as _base_get_config


def get_config():
    """Best params from sweep study 'raw_bci_cnnbilstm_sweep'."""
    c = _base_get_config()
    with c.unlocked():
        c.training.lr = 0.00042
        c.training.weight_decay = 0.013
        # ...
    c.lock()
    return c
```

The mapping from Optuna parameter names back to dotted config paths is recovered by re-running `define_space` against a recording trial in `_resolve_param_mapping`. If `define_space` does anything fancy (e.g. conditional `suggest_*` calls whose count differs from the returned dict size), the recovery falls back to identity mapping and the emitted config may be incomplete; keep `define_space` straight-line where possible.

Retrain the winner at full epoch budget with `python train.py --config <base>__best__<study>`. Now you get TensorBoard logs and a persistent checkpoint.

## Resuming and parallelism

- **Resumable:** re-invoking `python sweep.py --sweep <s> --study-name <name>` opens the same SQLite study (`load_if_exists=True`) and continues. New `--trials N` are *additional* trials on top of what is in the DB.
- **Sequential by design:** trials run with `n_jobs=1` because each trial saturates the GPU. Running two `sweep.py` processes against the same study DB works (Optuna handles the locking) but is rarely useful on a single-GPU box.
- **Cache hygiene:** see the cache keying gotcha in [data.md](data.md) - if you change `t_fork`, freq band, or window settings between studies, manually point `cfg.preprocessing.cache_path` somewhere fresh or regen the cache, since the cache path does not encode those.

## Existing sweeps

| Sweep | Base | Search space highlights |
|---|---|---|
| `raw_bci_cnnbilstm_sweep` | `raw_bci_cnnbilstm` | lr, wd, ls, bs, gnoise, chdrop, dropout, hidden, rnn_layers |
| `raw_cnnbilstm_sweep` | `raw_cnnbilstm` | same shape as above, Our5Class |
| `raw_bci_resnet18_sweep` | `raw_bci_resnet18` | optimizer + regularization sweep |
| `raw_bci_conformer_sweep` | `raw_bci_conformer` | attention/conv block hyperparams |
| `raw_bci_eegnet_sweep` | `raw_bci_eegnet` | F1, D, F2, kernel_length, sep_kernel + reg |
| `raw_eegnet_sweep` | `raw_eegnet` | EEGNet on Our5Class |
