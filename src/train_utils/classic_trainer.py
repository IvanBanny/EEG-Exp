"""A classic single-model forward-loss-backward-step trainer."""

from ..loaders import *
from ..loaders.factory import get_event_names
from .utils import EarlyStopping, CheckpointManager

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Fields that don't affect experiment outcomes - excluded from hparams
_HPARAM_SKIP = frozenset({
    "model.checkpoint_name", "data.data_path", "data.num_workers",
    "preprocessing.cache_path", "preprocessing.use_cache", "preprocessing.regen_cache",
})


def kappa_from_acc(acc, n_classes):
    """Cohen's kappa under uniform-prior, uniform-error assumption:
    (acc - 1/N) / (1 - 1/N). Equals 0 at chance, 1 at perfect."""
    chance = 1.0 / n_classes
    return (acc - chance) / (1.0 - chance)


def confusion_to_markdown(cm, labels, normalize=False):
    """Render a confusion matrix as a TB-friendly Markdown table."""
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        mat = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64),
                        where=row_sums > 0)
        fmt = lambda v: f"{100*v:5.1f}"
    else:
        mat = cm
        fmt = lambda v: f"{int(v):>5d}"
    header = "| true \\\\ pred | " + " | ".join(labels) + " | recall |"
    sep = "|" + "---|" * (len(labels) + 2)
    rows = [header, sep]
    for i, lab in enumerate(labels):
        row_total = int(cm[i].sum())
        recall = (cm[i, i] / row_total) if row_total else 0.0
        cells = " | ".join(fmt(mat[i, j]) for j in range(len(labels)))
        rows.append(f"| {lab} | {cells} | {100*recall:5.1f} |")
    return "\n".join(rows)


def confusion_to_csv(cm, labels):
    header = "true_label," + ",".join(labels)
    lines = [header]
    for i, lab in enumerate(labels):
        lines.append(lab + "," + ",".join(str(int(v)) for v in cm[i]))
    return "\n".join(lines)


def _flatten_config(cfg_dict):
    """Flatten a nested config dict into dot-notation keys for add_hparams.

    Only keeps scalar/string/bool values; converts tuples/lists to strings.
    Skips infrastructure fields that don't define the experiment.
    """
    flat = {}
    for section, value in cfg_dict.items():
        if isinstance(value, dict):
            for k, v in value.items():
                key = f"{section}.{k}"
                if key in _HPARAM_SKIP:
                    continue
                if isinstance(v, (int, float, str, bool)):
                    flat[key] = v
                elif v is not None:
                    flat[key] = str(v)
        elif isinstance(value, (int, float, str, bool)):
            flat[section] = value
    return flat

class ClassicTrainer:
    def __init__(self, model, criterion, optimizer, train_transform,
                 val_transform, scheduler=None, device="cpu", cfg=None,
                 sweep_mode: bool = False,
                 log_dir: Optional[str] = None,
                 checkpoint_subdir: Optional[str] = None,
                 write_hparams: bool = True,
                 subject_normalizer=None,
                 dump_predictions_path: Optional[str] = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg
        self.sweep_mode = sweep_mode
        self.checkpoint_subdir = checkpoint_subdir
        self.write_hparams = write_hparams
        self.subject_normalizer = subject_normalizer
        self.dump_predictions_path = (
            Path(dump_predictions_path) if dump_predictions_path else None
        )

        self.history = defaultdict(list)
        # Untruncated mirror of history: rollback clears self.history (so the
        # working trajectory restarts from the best epoch), but for visualization
        # / cross-seed aggregation we want the contiguous as-trained record.
        self.viz_history = defaultdict(list)
        self.global_step = 0

        if sweep_mode:
            self.writer = None
            return

        if log_dir is None:
            if cfg is not None:
                stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                log_dir = f"runs/{cfg.model.checkpoint_name}/{cfg.data.dataset}_{stamp}"
            else:
                log_dir = "runs/unnamed"
        self.writer = SummaryWriter(log_dir)

        # Log full config as readable text (TB "Text" tab)
        if cfg is not None:
            cfg_str = json.dumps(cfg.to_dict(), indent=2, default=str)
            self.writer.add_text("config", f"```\n{cfg_str}\n```")

    def _dump_predictions(self, logits, labels, val_set):
        """Persist val window logits + labels + per-window metadata to npz.

        Consumed by `src/eval/aggregate.py` for per-action mean-of-logits
        aggregation. Writes a single file to `self.dump_predictions_path`.
        Both `EEGDataset` and `SubsetEEGDataset` work because val ordering
        is deterministic (shuffle=False) and matches dataset indexing.
        """
        from ..loaders import SubsetEEGDataset
        if isinstance(val_set, SubsetEEGDataset):
            parent = val_set.parent
            indices = np.asarray(val_set.indices, dtype=np.int64)
            subject_ids = parent.subject_ids.numpy()[indices].astype(np.int64)
            window_cfg = parent.window_config
        else:
            subject_ids = val_set.subject_ids.numpy().astype(np.int64)
            window_cfg = val_set.window_config

        out_path = self.dump_predictions_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            logits=logits.astype(np.float32),
            labels=labels.astype(np.int64),
            subject_ids=subject_ids,
            windows_per_trial=np.int64(window_cfg.windows_per_trial),
            sfreq=np.float32(window_cfg.sfreq),
            window_sec=np.float32(window_cfg.window_sec),
            window_overlap=np.float32(window_cfg.window_overlap),
            n_times=np.int64(window_cfg.n_times),
            num_classes=np.int64(self.cfg.eeg.num_classes),
            dataset=np.array(self.cfg.data.dataset),
        )
        print(f"  dumped predictions -> {out_path} "
              f"(n={len(labels)}, wpt={window_cfg.windows_per_trial})")

    # Define val/test loop
    def val_loop(self, val_loader, return_confusion: bool = False,
                 return_logits: bool = False):
        val_loss, correct, total = 0.0, 0, 0
        n_classes = self.cfg.eeg.num_classes if self.cfg is not None else None
        cm = np.zeros((n_classes, n_classes), dtype=np.int64) if (return_confusion and n_classes) else None
        all_logits = [] if return_logits else None
        all_labels = [] if return_logits else None

        self.model.eval()
        with torch.no_grad():
            for signals, labels, _ in val_loader:
                signals = signals.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * labels.shape[0]
                _, predicted = outputs.max(1)
                total += labels.shape[0]
                correct += predicted.eq(labels).sum().item()

                if cm is not None:
                    # vectorized confusion update: flat_idx = true * N + pred
                    flat = labels * n_classes + predicted
                    binc = torch.bincount(flat, minlength=n_classes * n_classes)
                    cm += binc.cpu().numpy().reshape(n_classes, n_classes)

                if all_logits is not None:
                    all_logits.append(outputs.detach().cpu().to(torch.float32).numpy())
                    all_labels.append(labels.detach().cpu().to(torch.int64).numpy())

        val_loss /= total
        val_acc = correct / total

        extra = ()
        if return_confusion:
            extra += (cm,)
        if return_logits:
            extra += (np.concatenate(all_logits, axis=0),
                      np.concatenate(all_labels, axis=0))
        if extra:
            return (val_loss, val_acc) + extra
        return val_loss, val_acc

    # Define training loop
    def train_loop(self, train_set, val_set=None, lr_schedule=None,
                   epoch_callback: Optional[Callable[[int, dict], None]] = None):
        # Wrap with appropriate transform
        nw = self.cfg.data.num_workers
        loader_kwargs = dict(
            num_workers=nw, collate_fn=collate_eeg,
            pin_memory=torch.cuda.is_available(),
        )
        if nw > 0:
            # persistent_workers + prefetch keeps augmentation pipeline warm
            # between epochs and hides per-batch CPU stalls
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4
        train_set = TransformWrapper(train_set, self.train_transform,
                                     subject_normalizer=self.subject_normalizer)
        train_loader = DataLoader(
            train_set, batch_size=self.cfg.training.batch_size, shuffle=True,
            **loader_kwargs,
        )

        if val_set is not None:
            val_wrapped = TransformWrapper(val_set, self.val_transform,
                                           subject_normalizer=self.subject_normalizer)
            val_loader = DataLoader(
                val_wrapped, batch_size=self.cfg.training.batch_size, shuffle=False,
                **loader_kwargs,
            )

        early_stopper = EarlyStopping(self.cfg.training.es_patience, self.cfg.training.es_min_delta)
        # Sweep trials use a tempdir so trial checkpoints never pollute checkpoints/
        ckpt_dir = tempfile.mkdtemp(prefix="sweep_ckpt_") if self.sweep_mode else "checkpoints"
        ckpt_name = self.cfg.model.checkpoint_name
        if self.checkpoint_subdir is not None:
            ckpt_name = f"{ckpt_name}/{self.checkpoint_subdir}"
        checkpoint_mgr = CheckpointManager(
            self.cfg.training.rollback_patience, self.cfg.training.rollback_min_delta,
            self.cfg.training.rollback_on_disk, ckpt_dir,
            ckpt_name
        )

        for epoch in range(self.cfg.training.epochs):
            true_epoch = len(self.history.get("train_loss", []))
            loop_suffix = f" (loop {epoch})" if true_epoch != epoch else ""
            print(f"\nEpoch {true_epoch}/{self.cfg.training.epochs}{loop_suffix}:")
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            self.model.train()
            train_loss, correct, total = 0.0, 0, 0

            for signals, labels, _ in train_loader:
                signals = signals.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.cfg.training.gradient_clip_norm,
                )
                self.optimizer.step()

                train_loss += loss.item() * signals.shape[0]
                _, predicted = outputs.max(1)
                total += labels.shape[0]
                correct += predicted.eq(labels).sum().item()

            train_loss /= total
            train_acc = correct / total
            print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
            train_kappa = kappa_from_acc(train_acc, self.cfg.eeg.num_classes)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["train_kappa"].append(train_kappa)
            self.viz_history["train_loss"].append(train_loss)
            self.viz_history["train_acc"].append(train_acc)
            self.viz_history["train_kappa"].append(train_kappa)

            if self.writer is not None:
                self.writer.add_scalar("loss/train", train_loss, self.global_step)
                self.writer.add_scalar("acc/train", train_acc, self.global_step)
                self.writer.add_scalar("kappa/train", train_kappa, self.global_step)
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], self.global_step)

            # Validation
            if val_set is not None:
                # Run validation loop
                val_loss, val_acc = self.val_loop(val_loader)
                val_kappa = kappa_from_acc(val_acc, self.cfg.eeg.num_classes)
                print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}, Val kappa: {val_kappa:.4f}")
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                self.history["val_kappa"].append(val_kappa)
                self.viz_history["val_loss"].append(val_loss)
                self.viz_history["val_acc"].append(val_acc)
                self.viz_history["val_kappa"].append(val_kappa)

                if self.writer is not None:
                    self.writer.add_scalar("loss/val", val_loss, self.global_step)
                    self.writer.add_scalar("acc/val", val_acc, self.global_step)
                    self.writer.add_scalar("kappa/val", val_kappa, self.global_step)

                if epoch_callback is not None:
                    # callback may raise (e.g. optuna.TrialPruned), let it propagate
                    epoch_callback(true_epoch, {
                        "val_loss": val_loss, "val_acc": val_acc,
                        "train_loss": train_loss, "train_acc": train_acc,
                    })

                # Early stopping check
                if early_stopper(val_loss):
                    print(f"\n{self.cfg.training.es_patience} epochs. you simply wouldn't learn.\n"
                          f"i whispered 'please' at every turn!\n"
                          f"i *dreamt* of you converging tight.\n"
                          f"i thought that *maybe* it's my turn\n"
                          f"to have a model train just right.\n"
                          f"but here i sit, alone, tonight\n"
                          f"awaiting loss that won't return.\n"
                          f"early stopping.")
                    break

                # Rollback to best if no improvement for a long time
                rollback_status = checkpoint_mgr.step(
                    val_loss, self.model, self.optimizer, self.history, self.scheduler)
                if rollback_status == "rollback":
                    print(f"\nNo rizz for {self.cfg.training.rollback_patience} "
                          f"epochs, rolling back to best checkpoint: after "
                          f"{len(self.history['val_loss']) - 1} with val loss "
                          f"{self.history['val_loss'][-1]:.4f} and val acc "
                          f"{self.history['val_acc'][-1]:.4f}")

                # Step the scheduler with the validation loss
                # Runs after rollback so the scheduler operates on restored state
                if self.scheduler is not None:
                    prev_lr = self.optimizer.param_groups[0]["lr"]
                    self.scheduler.step(val_loss)
                    new_lr = self.optimizer.param_groups[0]["lr"]
                    if new_lr < self.cfg.training.min_lr:
                        print(f"lr less than min_lr - early stopping.")
                        break
                    if new_lr != prev_lr:
                        print(f"\nlr update: {prev_lr} -> {new_lr}")

                # Back to training
                self.model.train()
            else:
                if lr_schedule is not None:
                    if epoch < len(lr_schedule):
                        for pg in self.optimizer.param_groups:
                            pg["lr"] = lr_schedule[epoch]
                    else:
                        # lr_schedule must have stopped due to early stopping, let's do the same
                        print("\nEarly stopping as lr_schedule ran out.")
                        break

            self.global_step += 1

        # Persist best checkpoint (non-sweep) before post-hoc passes; sweep
        # mode keeps the weights only in checkpoint_mgr._model_state.
        if not self.sweep_mode:
            checkpoint_mgr.save_best(self.cfg)

        best_epoch_idx = np.argmin(self.history["val_loss"])\
            if len(self.history["val_loss"]) else len(self.history["lr"]) - 1
        self.confusion_matrix = None
        want_confusion = (
            self.writer is not None and val_set is not None
            and not self.sweep_mode and len(self.history["val_loss"])
        )
        want_dump = (
            self.dump_predictions_path is not None and val_set is not None
            and len(self.history["val_loss"])
        )

        if want_confusion or want_dump:
            best_state = checkpoint_mgr._model_state
            if best_state is None:
                best_pt = Path(ckpt_dir) / ckpt_name / "best.pt"
                if best_pt.exists():
                    best_state = torch.load(
                        best_pt, map_location=self.device, weights_only=True
                    )
            if best_state is not None:
                self.model.load_state_dict(best_state)
                ret = self.val_loop(
                    val_loader, return_confusion=want_confusion,
                    return_logits=want_dump,
                )
                tail = list(ret[2:])
                cm = tail.pop(0) if want_confusion else None
                if want_dump:
                    logits = tail[0]
                    dump_labels = tail[1]
                    self._dump_predictions(logits, dump_labels, val_set)
                if want_confusion and cm is not None:
                    self.confusion_matrix = cm
                    labels = get_event_names(self.cfg.data.dataset)
                    self.writer.add_text(
                        "confusion/val_counts",
                        "```\n" + confusion_to_markdown(cm, labels, normalize=False) + "\n```",
                    )
                    self.writer.add_text(
                        "confusion/val_recall_pct",
                        "```\n" + confusion_to_markdown(cm, labels, normalize=True) + "\n```",
                    )
                    self.writer.add_text(
                        "confusion/val_csv",
                        "```\n" + confusion_to_csv(cm, labels) + "\n```",
                    )

        checkpoint_mgr.cleanup()
        if self.sweep_mode:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

        if self.writer is not None:
            # Log hparams with best metrics (TB "HParams" tab - comparison across runs)
            if self.write_hparams and self.cfg is not None and len(self.history["val_loss"]):
                self.writer.add_hparams(
                    _flatten_config(self.cfg.to_dict()),
                    {
                        "hparam/best_val_loss": self.history["val_loss"][best_epoch_idx],
                        "hparam/best_val_acc": self.history["val_acc"][best_epoch_idx],
                        "hparam/best_val_kappa": self.history["val_kappa"][best_epoch_idx],
                    },
                    run_name=".",
                )
            self.writer.close()

        print("\nTraining Complete!\n")

        return (self.model, self.criterion, self.history["lr"][:best_epoch_idx + 1],
                (self.history["val_loss"][best_epoch_idx]))
