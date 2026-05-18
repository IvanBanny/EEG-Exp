"""Pooled training entry point.

Trains one model on the union of all subjects' training data and
validates on the union of their held-out splits. With
`cfg.training.num_runs > 1` the same architecture is trained N times
with seeds {cfg.seed, cfg.seed+1, ...}; each seed's TB log lands under
a parent dir, alongside a sibling `_agg/` carrying mean / std curves,
the mean confusion, and an hparams entry.

Aggregation reads per-seed scalars and checkpoints from disk so it
stays idempotent and picks up additional seeds added later. Datasets
and transforms are built once and reused across seeds.

Usage:
    python train.py --config raw_bci_eegencoder__winner_live
    # Extend an existing multi-seed run with more seeds:
    python train.py --config <name> --seeds 4272-4278 \\
        --parent-dir runs/<name>/<dataset>_<ts>_avg3
"""

import argparse
import importlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from src.loaders import (
    build_datasets, get_event_names,
    TransformWrapper, collate_eeg,
)
from src.models import build_model
from src.train_utils import ClassicTrainer, build_transforms
from src.train_utils.classic_trainer import (
    _flatten_config, confusion_to_markdown, confusion_to_csv,
)

_SCALAR_TAGS = ["loss/train", "loss/val", "acc/train", "acc/val",
                "kappa/train", "kappa/val"]


def _seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_seeds(spec):
    """Parse '4272-4278' (inclusive range) or '4272,4273,...' into list of ints."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def _read_seed_scalars(seed_dir, n_classes=None):
    """Read per-epoch scalar curves from a seed's TB events.

    Synthesises kappa from acc if the kappa scalars are absent (older
    runs predating kappa logging)."""
    ea = EventAccumulator(seed_dir, size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    out = {}
    for tag in _SCALAR_TAGS:
        if tag not in available:
            continue
        out[tag] = {s.step: s.value for s in ea.Scalars(tag)}
    if n_classes is not None:
        chance = 1.0 / n_classes
        for src, dst in (("acc/train", "kappa/train"), ("acc/val", "kappa/val")):
            if dst not in out and src in out:
                out[dst] = {step: (v - chance) / (1.0 - chance) for step, v in out[src].items()}
    return out


def _scalars_to_dense(curve, max_step):
    """{step: val} -> length max_step+1 array with NaN at missing steps."""
    out = np.full(max_step + 1, np.nan, dtype=np.float64)
    for step, val in curve.items():
        out[step] = val
    return out


def _compute_seed_confusion(seed, cfg, val_loader, device, n_classes):
    """Load checkpoints/<name>/seed_<seed>/best.pt, return confusion matrix."""
    ckpt = Path("checkpoints") / cfg.model.checkpoint_name / f"seed_{seed}" / "best.pt"
    if not ckpt.exists():
        return None
    model, _, _, _ = build_model(cfg, device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for signals, labels, _ in val_loader:
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            _, pred = model(signals).max(1)
            flat = labels * n_classes + pred
            binc = torch.bincount(flat, minlength=n_classes * n_classes)
            cm += binc.cpu().numpy().reshape(n_classes, n_classes)
    return cm


def _aggregate_from_disk(parent_dir, cfg, val_ds, val_transform, device,
                         subject_normalizer=None):
    """Rebuild <parent>/_agg by reading per-seed TB scalars + best.pt confusions.

    Idempotent: walks all seed_* subdirs of parent_dir, so picks up old and
    newly-added seeds uniformly. Returns (seeds, per_seed_best_loss).
    """
    parent = Path(parent_dir)
    seed_dirs = sorted([p for p in parent.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    seeds = [int(p.name.split("_")[1]) for p in seed_dirs]
    n_seeds = len(seeds)
    if not n_seeds:
        raise RuntimeError(f"no seed_* subdirs in {parent}")

    print(f"\nAggregating {n_seeds} seeds from {parent}")
    per_seed = [_read_seed_scalars(str(p), n_classes=cfg.eeg.num_classes) for p in seed_dirs]

    max_step = max(
        max(curve.keys()) for seed in per_seed for curve in seed.values() if curve
    )

    agg_dir = parent / "_agg"
    if agg_dir.exists():
        shutil.rmtree(agg_dir)
    writer = SummaryWriter(str(agg_dir))

    n_active = np.zeros(max_step + 1, dtype=np.int64)
    for tag in _SCALAR_TAGS:
        stack = []
        for seed in per_seed:
            if tag not in seed:
                continue
            stack.append(_scalars_to_dense(seed[tag], max_step))
        if not stack:
            continue
        mat = np.stack(stack)
        if tag == "loss/val":
            n_active = (~np.isnan(mat)).sum(axis=0)
        for step in range(max_step + 1):
            col = mat[:, step]
            if np.all(np.isnan(col)):
                continue
            writer.add_scalar(tag, float(np.nanmean(col)), step)
            writer.add_scalar(f"{tag}_std", float(np.nanstd(col)), step)
    for step in range(max_step + 1):
        writer.add_scalar("n_active_seeds", int(n_active[step]), step)

    # Per-seed best metrics from val_loss curves
    val_loss_curves = [s.get("loss/val", {}) for s in per_seed]
    val_acc_curves = [s.get("acc/val", {}) for s in per_seed]
    val_kappa_curves = [s.get("kappa/val", {}) for s in per_seed]
    per_seed_best_loss, per_seed_best_acc, per_seed_best_kappa = [], [], []
    for vl, va, vk in zip(val_loss_curves, val_acc_curves, val_kappa_curves):
        if not vl:
            continue
        best_step = min(vl, key=vl.get)
        per_seed_best_loss.append(float(vl[best_step]))
        per_seed_best_acc.append(float(va.get(best_step, np.nan)))
        per_seed_best_kappa.append(float(vk.get(best_step, np.nan)))
    mean_best_loss = float(np.mean(per_seed_best_loss))
    mean_best_acc = float(np.nanmean(per_seed_best_acc))
    mean_best_kappa = float(np.nanmean(per_seed_best_kappa))

    # Per-seed confusion (load each best.pt, val pass)
    n_classes = cfg.eeg.num_classes
    labels = get_event_names(cfg.data.dataset)
    val_loader = torch.utils.data.DataLoader(
        TransformWrapper(val_ds, val_transform,
                         subject_normalizer=subject_normalizer),
        batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, collate_fn=collate_eeg,
        pin_memory=torch.cuda.is_available(),
    )
    cms = []
    for seed in seeds:
        cm = _compute_seed_confusion(seed, cfg, val_loader, device, n_classes)
        if cm is not None:
            cms.append(cm)
    if cms:
        mean_cm = np.mean(np.stack(cms).astype(np.float64), axis=0)
        writer.add_text(
            "confusion/val_counts_mean",
            "```\n" + confusion_to_markdown(mean_cm, labels, normalize=False) + "\n```",
        )
        writer.add_text(
            "confusion/val_recall_pct_mean",
            "```\n" + confusion_to_markdown(mean_cm, labels, normalize=True) + "\n```",
        )
        writer.add_text(
            "confusion/val_csv_mean",
            "```\n" + confusion_to_csv(np.round(mean_cm).astype(np.int64), labels) + "\n```",
        )

    cfg_str = json.dumps(cfg.to_dict(), indent=2, default=str)
    writer.add_text("config", f"```\n{cfg_str}\n```")
    summary = {
        "seeds": seeds,
        "n_seeds": n_seeds,
        "per_seed_best_val_loss": per_seed_best_loss,
        "per_seed_best_val_acc": per_seed_best_acc,
        "per_seed_best_val_kappa": per_seed_best_kappa,
        "mean_best_val_loss": mean_best_loss,
        "mean_best_val_acc": mean_best_acc,
        "mean_best_val_kappa": mean_best_kappa,
    }
    writer.add_text("multi_seed_summary",
                    f"```\n{json.dumps(summary, indent=2)}\n```")
    writer.add_hparams(
        _flatten_config(cfg.to_dict()),
        {
            "hparam/best_val_loss": mean_best_loss,
            "hparam/best_val_acc": mean_best_acc,
            "hparam/best_val_kappa": mean_best_kappa,
        },
        run_name=".",
    )
    writer.close()
    print(f"  wrote {agg_dir} | n_seeds={n_seeds} max_step={max_step} "
          f"mean_best_val_acc={100*mean_best_acc:.2f}% kappa={mean_best_kappa:.4f}")
    return seeds, per_seed_best_loss


def _consolidate_best_checkpoint(cfg, seeds, per_seed_best_loss):
    """Copy the winning seed's best.{pt,yaml} up to checkpoints/<name>/."""
    if not seeds:
        return
    name = cfg.model.checkpoint_name
    winner_idx = int(np.argmin(per_seed_best_loss))
    winner_seed = seeds[winner_idx]

    src_dir = Path("checkpoints") / name / f"seed_{winner_seed}"
    dst_dir = Path("checkpoints") / name
    for fname in ("best.pt", "best.yaml"):
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)
    print(f"Consolidated best checkpoint from seed {winner_seed} "
          f"(val_loss {per_seed_best_loss[winner_idx]:.4f}) -> {dst_dir}/best.pt")


def main():
    parser = argparse.ArgumentParser(description="Train an EEG-MI model.")
    parser.add_argument("--config", required=True,
                        help="Config module name in configs/")
    parser.add_argument("--seeds", default=None,
                        help="Override seeds to run. '4272-4278' or '4272,4273'.")
    parser.add_argument("--parent-dir", default=None,
                        help="Override TB parent dir (to extend an existing experiment).")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip training; rebuild _agg from existing seed_* dirs in --parent-dir.")
    args = parser.parse_args()

    cfg = importlib.import_module(f"configs.{args.config}").get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, subject_normalizer = build_datasets(cfg)
    train_transform, val_transform = build_transforms(cfg, subject_normalizer)

    if args.aggregate_only:
        if args.parent_dir is None:
            raise SystemExit("--aggregate-only requires --parent-dir")
        all_seeds, per_seed_best_loss = _aggregate_from_disk(
            args.parent_dir, cfg, val_ds, val_transform, device,
            subject_normalizer=subject_normalizer,
        )
        _consolidate_best_checkpoint(cfg, all_seeds, per_seed_best_loss)
        return

    num_runs = int(getattr(cfg.training, "num_runs", 1))
    base_seed = int(cfg.seed)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Single-run path.
    if num_runs == 1 and args.seeds is None:
        _seed_all(base_seed)
        model, criterion, optimizer, scheduler = build_model(cfg, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{cfg.model.arch} | {n_params:,} params | {device}\n")
        trainer = ClassicTrainer(
            model, criterion, optimizer,
            train_transform, val_transform,
            scheduler, device, cfg,
            subject_normalizer=subject_normalizer,
        )
        trainer.train_loop(train_set=train_ds, val_set=val_ds)
        return

    if args.seeds is not None:
        seeds = _parse_seeds(args.seeds)
    else:
        seeds = [base_seed + i for i in range(num_runs)]

    parent_log_dir = args.parent_dir or (
        f"runs/{cfg.model.checkpoint_name}/{cfg.data.dataset}_{stamp}_avg{len(seeds)}"
    )

    for i, seed in enumerate(seeds):
        print(f"\n{'='*64}\nRun {i+1}/{len(seeds)} (seed={seed})\n{'='*64}")
        _seed_all(seed)
        model, criterion, optimizer, scheduler = build_model(cfg, device)
        if i == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"\n{cfg.model.arch} | {n_params:,} params | {device}\n")
        trainer = ClassicTrainer(
            model, criterion, optimizer,
            train_transform, val_transform,
            scheduler, device, cfg,
            log_dir=f"{parent_log_dir}/seed_{seed}",
            checkpoint_subdir=f"seed_{seed}",
            write_hparams=False,
            subject_normalizer=subject_normalizer,
        )
        trainer.train_loop(train_set=train_ds, val_set=val_ds)

    # Aggregate from disk so old and newly-added seeds are unified.
    all_seeds, per_seed_best_loss = _aggregate_from_disk(
        parent_log_dir, cfg, val_ds, val_transform, device,
        subject_normalizer=subject_normalizer,
    )
    _consolidate_best_checkpoint(cfg, all_seeds, per_seed_best_loss)


if __name__ == "__main__":
    main()
