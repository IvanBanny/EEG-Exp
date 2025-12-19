"""A classic single-model forward-loss-backward-step trainer."""

from ..loaders import *
from .utils import EarlyStopping, CheckpointManager

from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class ClassicTrainer:
    def __init__(self, model, criterion, optimizer, train_transform,
                 val_transform, scheduler=None, device="cpu", cfg=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg

        self.history = defaultdict(list)

    # Define val/test loop
    def val_loop(self, val_set):
        # Wrap with appropriate transform
        val_set = TransformWrapper(val_set, self.val_transform)
        val_loader = DataLoader(
            val_set, batch_size=self.cfg.training.batch_size, shuffle=False,
            num_workers=self.cfg.data.num_workers, collate_fn=collate_spectrograms,
            pin_memory=torch.cuda.is_available()
        )

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
    def train_loop(self, train_set, val_set=None, lr_schedule=None):
        # Wrap with appropriate transform
        train_set = TransformWrapper(train_set, self.train_transform)
        train_loader = DataLoader(
            train_set, batch_size=self.cfg.training.batch_size, shuffle=True,
            num_workers=self.cfg.data.num_workers, collate_fn=collate_spectrograms,
            pin_memory=torch.cuda.is_available()
        )

        # sample, label, lengths = next(iter(train_loader))
        # print(f"Sample stats: min={sample.min():.4f}, max={sample.max():.4f}, "
        #       f"mean={sample.mean():.4f}, std={sample.std():.4f}")
        # print(f"Any NaN: {sample.isnan().any()}, Any Inf: {sample.isinf().any()}")
        # print(f"Label distribution: {torch.bincount(label)}")

        early_stopper = EarlyStopping(self.cfg.training.es_patience, self.cfg.training.es_min_delta)
        checkpoint_mgr = CheckpointManager(
            self.cfg.training.rollback_patience, self.cfg.training.rollback_min_delta,
            self.cfg.training.rollback_on_disk, "checkpoints",
            self.cfg.model.checkpoint_name
        )

        for epoch in range(self.cfg.training.epochs):
            true_epoch = len(self.history.get('train_loss', []))
            loop_suffix = f" (loop {epoch})" if true_epoch != epoch else ''
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

            # Validation
            if val_set is not None:
                # Run validation loop
                val_loss, val_acc = self.val_loop(val_set)
                print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                # Early stopping check
                if early_stopper(val_loss):
                    print(f"\n{self.cfg.training.es_patience} epochs. you simply wouldn't learn.\n"
                          f"i whispered 'please' at every turn!\n"
                          f"i *dreamed* of you converging right.\n"
                          f"i thought that *maybe* it's my turn\n"
                          f"to have a model train just right.\n"
                          f"but here i sit alone tonight\n"
                          f"awaiting loss that won't return.\n"
                          f"early stopping.")
                    break

                # Step the scheduler with the validation loss
                if self.scheduler is not None:
                    prev_lr = self.optimizer.param_groups[0]["lr"]
                    self.scheduler.step(val_loss)
                    new_lr = self.optimizer.param_groups[0]["lr"]
                    if new_lr < self.cfg.training.min_lr:
                        print(f"lr less than min_lr - early stopping.")
                        break
                    if new_lr != prev_lr:
                        print(f"\nlr update: {prev_lr} -> {new_lr}")

                # Rollback to best if no improvement for a long time
                if checkpoint_mgr.step(val_loss, self.model, self.optimizer, self.history) == "rollback":
                    print(f"\nNo improvement for {self.cfg.training.rollback_patience} "
                          f"epochs, rolling back to best checkpoint after "
                          f"{len(self.history['val_loss']) - 1} with val loss of "
                          f"{self.history['val_loss'][-1]:.4f}")

                # Back to training
                self.model.train()
            else:
                if lr_schedule is not None:
                    if epoch < len(lr_schedule):
                        for pg in self.optimizer.param_groups:
                            pg["lr"] = self.cfg.training.lr if lr_schedule is None else lr_schedule[epoch]
                    else:
                        # lr_schedule must have stopped due to early stopping, let's do the same
                        print("\nEarly stopping as lr_schedule ran out.")
                        break

        checkpoint_mgr.save_best(self.cfg)  # Save the best model and history to /checkpoints
        checkpoint_mgr.cleanup()  # Cleanup the rollback checkpoint if on disk

        print("\nTraining Complete!\n")

        best_epoch_idx = np.argmin(self.history["val_loss"])\
            if len(self.history["val_loss"]) else len(self.history["lr"]) - 1

        return (self.model, self.criterion, self.history["lr"][:best_epoch_idx + 1],
                (self.history["val_loss"][best_epoch_idx]))
