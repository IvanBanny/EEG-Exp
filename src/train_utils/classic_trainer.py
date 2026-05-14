"""A classic single-model forward-loss-backward-step trainer."""

from ..loaders import *
from .utils import EarlyStopping, CheckpointManager

from collections import defaultdict
from datetime import datetime
import json
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
                 sweep_mode: bool = False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg
        self.sweep_mode = sweep_mode

        self.history = defaultdict(list)
        self.global_step = 0

        if sweep_mode:
            self.writer = None
            return

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

    # Define val/test loop
    def val_loop(self, val_loader):
        val_loss, correct, total = 0.0, 0, 0

        self.model.eval()
        with torch.no_grad():
            for signals, labels, lengths in val_loader:  # TODO: DO SOMETHING ABOUT LENGTHS?
                signals = signals.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * labels.shape[0]
                _, predicted = outputs.max(1)
                total += labels.shape[0]
                correct += predicted.eq(labels).sum().item()

        val_loss /= total
        val_acc = correct / total

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
        train_set = TransformWrapper(train_set, self.train_transform)
        train_loader = DataLoader(
            train_set, batch_size=self.cfg.training.batch_size, shuffle=True,
            **loader_kwargs,
        )

        if val_set is not None:
            val_wrapped = TransformWrapper(val_set, self.val_transform)
            val_loader = DataLoader(
                val_wrapped, batch_size=self.cfg.training.batch_size, shuffle=False,
                **loader_kwargs,
            )

        early_stopper = EarlyStopping(self.cfg.training.es_patience, self.cfg.training.es_min_delta)
        # Sweep trials use a tempdir so trial checkpoints never pollute checkpoints/
        ckpt_dir = tempfile.mkdtemp(prefix="sweep_ckpt_") if self.sweep_mode else "checkpoints"
        checkpoint_mgr = CheckpointManager(
            self.cfg.training.rollback_patience, self.cfg.training.rollback_min_delta,
            self.cfg.training.rollback_on_disk, ckpt_dir,
            self.cfg.model.checkpoint_name
        )

        for epoch in range(self.cfg.training.epochs):
            true_epoch = len(self.history.get("train_loss", []))
            loop_suffix = f" (loop {epoch})" if true_epoch != epoch else ""
            print(f"\nEpoch {true_epoch}/{self.cfg.training.epochs}{loop_suffix}:")
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            self.model.train()
            train_loss, correct, total = 0.0, 0, 0

            for signals, labels, lengths in train_loader:  # TODO: DO SOMETHING ABOUT LENGTHS
                # signals is usually `batch_size x num_channels x H x W` (after STFT/VMD)
                # or `batch_size x num_channels x num_samples`
                # depending on which transforms you use in the dataset adapters
                # where the last dim (W / num_samples) must be the same length for all
                # so it must be padded in most cases and the real len is in the lengths tensor
                # An example: signals shape `32 x 22 x 33 x 48`, labels shape `32`, lengths shape `32`
                signals = signals.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                loss.backward()

                # # Debug: check gradients
                # total_grad = 0
                # for p in self.model.parameters():
                #     if p.grad is not None:
                #         total_grad += p.grad.abs().mean().item()
                # if signals.shape[0] == self.cfg.training.batch_size:  # first full batch
                #     print(f"Avg gradient magnitude: {total_grad:.8f}")

                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         max_norm=self.cfg.training.gradient_clip_norm)  # Gradient clipping
                self.optimizer.step()

                train_loss += loss.item() * signals.shape[0]
                _, predicted = outputs.max(1)
                total += labels.shape[0]
                correct += predicted.eq(labels).sum().item()

            train_loss /= total
            train_acc = correct / total
            print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            if self.writer is not None:
                self.writer.add_scalar("loss/train", train_loss, self.global_step)
                self.writer.add_scalar("acc/train", train_acc, self.global_step)
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], self.global_step)

            # Validation
            if val_set is not None:
                # Run validation loop
                val_loss, val_acc = self.val_loop(val_loader)
                print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                if self.writer is not None:
                    self.writer.add_scalar("loss/val", val_loss, self.global_step)
                    self.writer.add_scalar("acc/val", val_acc, self.global_step)

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

        if self.sweep_mode:
            checkpoint_mgr.cleanup()
            shutil.rmtree(ckpt_dir, ignore_errors=True)
        else:
            checkpoint_mgr.save_best(self.cfg)  # Save best model + config to /checkpoints
            checkpoint_mgr.cleanup()  # Cleanup the rollback checkpoint if on disk

        best_epoch_idx = np.argmin(self.history["val_loss"])\
            if len(self.history["val_loss"]) else len(self.history["lr"]) - 1
        if self.writer is not None:
            # Log hparams with best metrics (TB "HParams" tab - comparison across runs)
            if self.cfg is not None and len(self.history["val_loss"]):
                self.writer.add_hparams(
                    _flatten_config(self.cfg.to_dict()),
                    {
                        "hparam/best_val_loss": self.history["val_loss"][best_epoch_idx],
                        "hparam/best_val_acc": self.history["val_acc"][best_epoch_idx],
                    },
                    run_name=".",
                )
            self.writer.close()

        print("\nTraining Complete!\n")

        return (self.model, self.criterion, self.history["lr"][:best_epoch_idx + 1],
                (self.history["val_loss"][best_epoch_idx]))
