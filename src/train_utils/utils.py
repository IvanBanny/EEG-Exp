"""Training utils."""

import copy
from pathlib import Path
import torch


class EarlyStopping:
    """Manages early stopping based on val loss."""
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

class CheckpointManager:
    """Stores and restores the best checkpoint based on val loss."""
    def __init__(self, patience=5, min_delta=0.0, save_to_disk=False, checkpoint_dir="checkpoints", checkpoint_name="best"):
        self.patience = patience
        self.min_delta=min_delta
        self.save_to_disk=save_to_disk
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name=checkpoint_name

        self.best_loss = None
        self.counter = 0

        self._checkpoint_dir_pth = Path(checkpoint_dir) / self.checkpoint_name
        self._checkpoint_pth = self._checkpoint_dir_pth / "rollback.pt"

        self._model_state = None
        self._optimizer_state = None
        self._history_snapshot = None

    def step(self, val_loss, model, optimizer, history, scheduler=None):
        """Call after each validation.

        Returns:
            "improved" - new best, checkpoint saved.
            "patience" - no improvement, but still waiting.
            "rollback" - patience exhausted, states restored.
        """
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self._save(model, optimizer, history, scheduler)
            return "improved"

        self.counter += 1

        if self.counter >= self.patience:
            self._restore(model, optimizer, history, scheduler)
            self.counter = 0
            return "rollback"

        return "patience"

    def _save(self, model, optimizer, history, scheduler=None):
        if self.save_to_disk:
            self._checkpoint_dir_pth.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": dict(history)
            }, self._checkpoint_pth)
        else:
            self._model_state = copy.deepcopy(model.state_dict())
            self._optimizer_state = copy.deepcopy(optimizer.state_dict())
            self._history_snapshot = copy.deepcopy(dict(history))

    def _restore(self, model, optimizer, history, scheduler=None):
        # Scheduler is intentionally NOT restored - it tracks overall training
        # progress across rollback cycles, so its LR reductions accumulate
        # We only restore model weights, optimizer momentum, and history
        current_lrs = [pg["lr"] for pg in optimizer.param_groups]

        if self.save_to_disk:
            if not self._checkpoint_pth.exists():
                return
            device = next(model.parameters()).device
            checkpoint_obj = torch.load(self._checkpoint_pth, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint_obj["model"])
            optimizer.load_state_dict(checkpoint_obj["optimizer"])
            history_snapshot = checkpoint_obj["history"]
        else:
            if self._model_state is None:
                return
            model.load_state_dict(self._model_state)
            optimizer.load_state_dict(self._optimizer_state)
            history_snapshot = self._history_snapshot

        # optimizer.load_state_dict overwrites LR with the checkpointed value;
        # sync it back to whatever the scheduler currently has
        for pg, lr in zip(optimizer.param_groups, current_lrs):
            pg["lr"] = lr

        # Rollback consumed the scheduler's patience - prime it so the
        # scheduler.step() call at the end of THIS epoch triggers a reduction
        # ReduceLROnPlateau fires when num_bad_epochs > patience (strict >),
        # and step() increments before checking, so we need patience here
        if scheduler is not None and hasattr(scheduler, "num_bad_epochs"):
            scheduler.num_bad_epochs = scheduler.patience

        # Restore history
        history.clear()
        history.update(copy.deepcopy(history_snapshot))

    def save_best(self, cfg=None):
        """Save the best model state to .pt and config to .yaml."""
        self._checkpoint_dir_pth.mkdir(parents=True, exist_ok=True)

        # Get model state (from disk or memory)
        if self.save_to_disk:
            if not self._checkpoint_pth.exists():
                return
            checkpoint_obj = torch.load(self._checkpoint_pth, weights_only=False)
            model_state = checkpoint_obj["model"]
        else:
            if self._model_state is None:
                return
            model_state = self._model_state

        # Save model
        model_path = self._checkpoint_dir_pth / "best.pt"
        torch.save(model_state, model_path)

        # Save config as yaml
        if cfg is not None:
            config_path = self._checkpoint_dir_pth / "best.yaml"
            with open(config_path, 'w') as f:
                f.write(cfg.to_yaml())

    def cleanup(self):
        """Remove checkpoint file if using disk storage."""
        if self.save_to_disk:
            self._checkpoint_pth.unlink(missing_ok=True)