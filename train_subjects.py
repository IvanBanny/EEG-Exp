"""Per-subject training entry point.

For each `(subject, seed)` cell, trains one model on that subject's
training data only and evaluates on its held-out split (`last_poly5`
for Our5Class, `last_run` for BCI 2a). Each cell writes a `result.json`
(best metrics + per-cell metadata), a `predictions.npz` (per-window
val logits + labels + per-window metadata), and a `best.pt`.

Output layout:
    runs/<config>/subject_<N>/seed_<S>/
        events.out.tfevents.*
        predictions.npz
        result.json
    checkpoints/<config>/subject_<N>/seed_<S>/
        best.pt

Idempotent - cells with an existing `result.json` are skipped. Post-hoc
aggregation: `python -m src.eval.aggregate --runs-dir runs/<config>`.

Usage:
    python train_subjects.py --config raw_our5_eegencoder__winner_live \\
        --subjects 1-14 --seeds 4269-4271
    python train_subjects.py --config raw_bci_eegnet__winner_live \\
        --subjects 1-9 --seeds 4269-4271 --max-epochs 500 --no-early-stop
"""

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.loaders import build_datasets
from src.models import build_model
from src.train_utils import ClassicTrainer, build_transforms


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_range(spec: str) -> list[int]:
    """Parse '1-9' (inclusive) or '1,3,5' into a list of ints."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def _override_cfg(cfg, config_name: str, subject: int, seed: int,
                  max_epochs: int | None, disable_early_stop: bool):
    """Pin a per-cell config: subject filter, seed, checkpoint name."""
    with cfg.unlocked():
        cfg.seed = int(seed)
        cfg.data.subjects = [int(subject)]
        cfg.model.checkpoint_name = f"{config_name}/subject_{subject}"
        if max_epochs is not None:
            cfg.training.epochs = int(max_epochs)
        if disable_early_stop:
            # Disable ES / rollback / LR plateau so the loop runs to
            # the full epoch budget. Used for ceiling tests on BCI 2a.
            big = cfg.training.epochs + 1
            cfg.training.es_patience = big
            cfg.training.rollback_patience = big
            cfg.training.lr_patience = big
    return cfg


def _train_cell(config_name: str, subject: int, seed: int,
                max_epochs: int | None, disable_early_stop: bool,
                device: torch.device) -> dict:
    cell_dir = Path("runs") / config_name / f"subject_{subject}" / f"seed_{seed}"
    result_path = cell_dir / "result.json"
    if result_path.exists():
        print(f"  skip (exists): {result_path}")
        with open(result_path) as f:
            return json.load(f)
    cell_dir.mkdir(parents=True, exist_ok=True)

    cfg_module = importlib.import_module(f"configs.{config_name}")
    cfg = cfg_module.get_config()
    cfg = _override_cfg(cfg, config_name, subject, seed,
                        max_epochs, disable_early_stop)

    _seed_all(seed)
    t0 = time.time()
    train_ds, val_ds, normalizer = build_datasets(cfg)
    build_secs = time.time() - t0
    input_shape = tuple(train_ds[0][0].shape)
    print(f"  built in {build_secs:.1f}s | train={len(train_ds)} "
          f"val={len(val_ds)} input={input_shape}")

    model, criterion, optimizer, scheduler = build_model(cfg, device)
    train_t, val_t = build_transforms(cfg, normalizer)

    pred_path = cell_dir / "predictions.npz"
    trainer = ClassicTrainer(
        model, criterion, optimizer, train_t, val_t, scheduler, device, cfg,
        log_dir=str(cell_dir),
        checkpoint_subdir=f"seed_{seed}",
        write_hparams=False,
        subject_normalizer=normalizer,
        dump_predictions_path=str(pred_path),
    )

    t0 = time.time()
    trainer.train_loop(train_set=train_ds, val_set=val_ds)
    train_secs = time.time() - t0

    val_loss = trainer.viz_history.get("val_loss", []) or trainer.history.get("val_loss", [])
    val_acc = trainer.viz_history.get("val_acc", []) or trainer.history.get("val_acc", [])
    val_kappa = trainer.viz_history.get("val_kappa", []) or trainer.history.get("val_kappa", [])
    n_epochs = len(val_loss)
    best_loss_idx = int(np.argmin(val_loss)) if n_epochs else -1
    best_acc_idx = int(np.argmax(val_acc)) if n_epochs else -1

    result = {
        "config": config_name,
        "subject": int(subject),
        "seed": int(seed),
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "input_shape": list(input_shape),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "epochs_run": int(n_epochs),
        "best_loss_epoch": int(best_loss_idx),
        "best_val_loss": float(val_loss[best_loss_idx]) if n_epochs else None,
        "val_acc_at_best_loss": float(val_acc[best_loss_idx]) if n_epochs else None,
        "val_kappa_at_best_loss": float(val_kappa[best_loss_idx]) if n_epochs else None,
        "best_acc_epoch": int(best_acc_idx),
        "best_val_acc": float(val_acc[best_acc_idx]) if n_epochs else None,
        "best_val_kappa": float(val_kappa[best_acc_idx]) if n_epochs else None,
        "val_loss_at_best_acc": float(val_loss[best_acc_idx]) if n_epochs else None,
        "build_secs": float(build_secs),
        "train_secs": float(train_secs),
        "predictions_path": str(pred_path),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  -> best_acc={result['best_val_acc']:.4f} "
          f"kappa={result['best_val_kappa']:.4f} "
          f"(epoch {best_acc_idx}/{n_epochs}) | train_secs={train_secs:.1f}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train one model per subject from a winner config."
    )
    parser.add_argument("--config", required=True,
                        help="Config module name under configs/.")
    parser.add_argument("--subjects", required=True,
                        help="'1-14' or '1,3,5'.")
    parser.add_argument("--seeds", required=True,
                        help="'4269-4271' or '4269,4270,4271'.")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override cfg.training.epochs.")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable ES / rollback / LR plateau "
                             "(runs the full epoch budget).")
    args = parser.parse_args()

    subjects = _parse_range(args.subjects)
    seeds = _parse_range(args.seeds)
    cells = [(s, sd) for s in subjects for sd in seeds]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Config '{args.config}': {len(cells)} cells "
          f"(subjects={subjects}, seeds={seeds}) "
          f"max_epochs={args.max_epochs} no_es={args.no_early_stop} "
          f"device={device}")
    t0 = time.time()
    for i, (subj, seed) in enumerate(cells, 1):
        print(f"\n=== [{i}/{len(cells)}] config={args.config} "
              f"subject={subj} seed={seed} ===")
        _train_cell(args.config, subj, seed,
                    args.max_epochs, args.no_early_stop, device)
    print(f"\nConfig '{args.config}' done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
