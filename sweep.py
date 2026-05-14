"""Hyperparameter sweep CLI.

Usage:
    python sweep.py --sweep raw_bci_cnnbilstm_sweep --trials 100
    python sweep.py --sweep raw_bci_cnnbilstm_sweep --smoke
    python sweep.py --sweep <name> --trials 200 --study-name lr_v2
"""

import argparse
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

from src.loaders import (
    build_datasets,
    Compose, ClipOutliers, LogCompress, ZScoreNormalize,
    GaussianNoise, RandomScale, TimeShift, ChannelDropout,
)
from src.models import build_model
from src.train_utils import ClassicTrainer

_FORBIDDEN_PREFIXES = ("data.", "preprocessing.")
_STUDY_DIR = Path("sweeps/.studies")


def _apply_overrides(cfg, overrides: dict):
    """Apply dotted-path overrides to a (possibly locked) ConfigDict.

    Re-locks the config before returning to mirror the contract of base configs.
    """
    with cfg.unlocked():
        for dotted, value in overrides.items():
            parts = dotted.split(".")
            node = cfg
            for p in parts[:-1]:
                node = getattr(node, p)
            setattr(node, parts[-1], value)
    cfg.lock()
    return cfg


def _validate_overrides(overrides: dict):
    bad = [k for k in overrides if k.startswith(_FORBIDDEN_PREFIXES)]
    if bad:
        raise ValueError(
            f"Sweep returned forbidden override keys {bad}. "
            f"data.* and preprocessing.* would invalidate the cached datasets "
            f"that are built once outside the objective."
        )


def _build_transforms(cfg):
    """Mirror train.py transform construction exactly."""
    base = [ClipOutliers(sigma=cfg.regularization.clip_sigma)]
    if cfg.preprocessing.representation == "stft":
        base.append(LogCompress())
    base.append(ZScoreNormalize())
    train_transform = Compose(base + [
        GaussianNoise(std=cfg.regularization.gaussian_std),
        RandomScale(scale_fork=cfg.regularization.scale_fork),
        TimeShift(max_shift=cfg.regularization.max_shift),
        ChannelDropout(p=cfg.regularization.channel_dropout),
    ])
    val_transform = Compose(list(base))
    return train_transform, val_transform


def _seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_objective(sweep_module, base_seed, max_epochs, train_ds, val_ds, device, study_name):
    base_cfg_name = sweep_module.BASE_CONFIG
    metric = sweep_module.METRIC
    direction = sweep_module.DIRECTION

    def objective(trial: optuna.Trial) -> float:
        overrides = sweep_module.define_space(trial)
        _validate_overrides(overrides)

        cfg_module = importlib.import_module(f"configs.{base_cfg_name}")
        cfg = cfg_module.get_config()
        cfg = _apply_overrides(cfg, overrides)

        with cfg.unlocked():
            cfg.training.epochs = max_epochs
            cfg.seed = base_seed + trial.number
            cfg.model.checkpoint_name = f"{study_name}_t{trial.number}"
            # In-memory datasets + small models => DataLoader worker spawn
            # dominates per-trial cost. Drop to a single producer.
            cfg.data.num_workers = 2
        cfg.lock()

        _seed_all(cfg.seed)

        try:
            model, criterion, optimizer, scheduler = build_model(cfg, device)
            train_transform, val_transform = _build_transforms(cfg)

            trainer = ClassicTrainer(
                model, criterion, optimizer,
                train_transform, val_transform,
                scheduler, device, cfg, sweep_mode=True,
            )

            def epoch_callback(epoch: int, metrics: dict):
                trial.report(metrics[metric], step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            trainer.train_loop(train_set=train_ds, val_set=val_ds,
                               epoch_callback=epoch_callback)

            history_vals = trainer.history.get(metric, [])
            if not history_vals:
                raise optuna.TrialPruned()
            return max(history_vals) if direction == "maximize" else min(history_vals)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective


def _load_sweep_module(name: str):
    """Load a sweep module by name. Accepts 'foo' or 'sweeps.foo'."""
    short = name[len("sweeps."):] if name.startswith("sweeps.") else name
    return importlib.import_module(f"sweeps.{short}")


def _write_best_config(base_cfg_name: str, study_name: str, best_params: dict,
                       sweep_module) -> Path:
    """Emit configs/<base>__best__<study>.py that bakes the best params in."""
    out_path = Path("configs") / f"{base_cfg_name}__best__{study_name}.py"

    # The sweep file's define_space maps trial-suggest names to dotted config paths.
    # We re-derive that mapping by intercepting suggest_* calls so we can convert
    # Optuna's best_params (keyed by suggest names) into config dotted-path overrides.
    name_to_dotted = _resolve_param_mapping(sweep_module)

    overrides = {}
    for suggest_name, value in best_params.items():
        if suggest_name not in name_to_dotted:
            # parameter not surfaced via define_space (should not happen); skip
            continue
        overrides[name_to_dotted[suggest_name]] = value

    lines = [
        f'"""Best config from study {study_name!r} (base: {base_cfg_name})."""',
        "",
        f"from .{base_cfg_name} import get_config as _base_get_config",
        "",
        "",
        "def get_config():",
        f'    """Best params from sweep study {study_name!r}."""',
        "    c = _base_get_config()",
        "    with c.unlocked():",
    ]
    for dotted, value in overrides.items():
        lines.append(f"        c.{dotted} = {value!r}")
    lines.append("    c.lock()")
    lines.append("    return c")
    lines.append("")
    out_path.write_text("\n".join(lines))
    return out_path


def _resolve_param_mapping(sweep_module) -> dict:
    """Recover {suggest_name: dotted_config_path} by running define_space against a recording trial."""
    class _RecordingTrial:
        def __init__(self):
            self.last_name = None

        def _record(self, name, default):
            self.last_name = name
            return default

        def suggest_float(self, name, low, high, log=False, step=None):
            return self._record(name, low)

        def suggest_int(self, name, low, high, log=False, step=1):
            return self._record(name, low)

        def suggest_categorical(self, name, choices):
            return self._record(name, choices[0])

    # Run define_space and capture order of suggest calls vs returned dict keys.
    # The cleanest mapping: call once, but we need name <-> dotted-path pairs.
    # Re-run define_space and pair by call order using a mutable list.
    calls = []

    class _OrderedTrial(_RecordingTrial):
        def _record(self, name, default):
            calls.append(name)
            return default

    rec = _OrderedTrial()
    overrides = sweep_module.define_space(rec)
    dotted_paths = list(overrides.keys())
    if len(dotted_paths) != len(calls):
        # define_space did something fancy (e.g. conditional suggestions)
        # Fall back to identity mapping where possible
        return {name: name for name in calls}
    return dict(zip(calls, dotted_paths))


def main():
    parser = argparse.ArgumentParser(description="Run an Optuna sweep.")
    parser.add_argument("--sweep", required=True, help="Module name under sweeps/")
    parser.add_argument("--trials", type=int, default=100,
                        help="Number of trials to run this invocation.")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Wall-clock seconds budget for this invocation; trials stop once exceeded.")
    parser.add_argument("--study-name", default=None,
                        help="Optuna study name (defaults to --sweep). Used as DB filename too.")
    parser.add_argument("--smoke", action="store_true",
                        help="2 trials x 3 epochs plumbing check, written to <study>_smoke.")
    parser.add_argument("--min-resource", type=int, default=5)
    parser.add_argument("--reduction-factor", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None,
                        help="TPE seed and trial seed offset (defaults to base config seed).")
    args = parser.parse_args()

    sweep_module = _load_sweep_module(args.sweep)

    base_cfg_name = sweep_module.BASE_CONFIG
    base_cfg_module = importlib.import_module(f"configs.{base_cfg_name}")
    base_cfg = base_cfg_module.get_config()

    seed = args.seed if args.seed is not None else base_cfg.seed
    max_epochs = 3 if args.smoke else sweep_module.MAX_EPOCHS
    n_trials = 2 if args.smoke else args.trials

    study_name = args.study_name or args.sweep
    if args.smoke:
        study_name = f"{study_name}_smoke"

    _STUDY_DIR.mkdir(parents=True, exist_ok=True)
    storage_path = _STUDY_DIR / f"{study_name}.db"
    storage_url = f"sqlite:///{storage_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build datasets ONCE - sweep is forbidden from overriding data/preprocessing,
    # so the cached datasets are valid across all trials in the study.
    _seed_all(seed)
    print(f"\nBuilding datasets (base config: {base_cfg_name})...")
    train_ds, val_ds = build_datasets(base_cfg)

    sampler = TPESampler(multivariate=True, group=True, seed=seed)
    pruner = SuccessiveHalvingPruner(
        min_resource=args.min_resource,
        reduction_factor=args.reduction_factor,
        min_early_stopping_rate=0,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction=sweep_module.DIRECTION,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    objective = _make_objective(
        sweep_module, base_seed=seed, max_epochs=max_epochs,
        train_ds=train_ds, val_ds=val_ds, device=device, study_name=study_name,
    )

    print(f"\nStudy: {study_name} | trials_to_run: {n_trials} | max_epochs: {max_epochs}")
    print(f"Storage: {storage_url}")
    print(f"Existing trials: {len(study.trials)}\n")

    try:
        study.optimize(objective, n_trials=n_trials, timeout=args.timeout,
                       n_jobs=1, gc_after_trial=True)
    except KeyboardInterrupt:
        print("\nInterrupted - reporting partial study.")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print("\n" + "=" * 60)
    print(f"Study {study_name} summary")
    print(f"  completed: {len(completed)} | pruned: {len(pruned)} | failed: {len(failed)}")
    if completed:
        best = study.best_trial
        print(f"  best trial: #{best.number}  value={best.value:.4f}")
        print(f"  best params:")
        for k, v in best.params.items():
            print(f"    {k}: {v}")

        out_path = _write_best_config(base_cfg_name, study_name, best.params, sweep_module)
        print(f"\nWrote winner config: {out_path}")
        print(f"Retrain with: python train.py --config {out_path.stem}")
    else:
        print("  no completed trials yet")
    print("=" * 60)


if __name__ == "__main__":
    main()
