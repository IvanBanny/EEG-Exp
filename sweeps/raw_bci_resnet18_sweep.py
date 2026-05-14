"""1D ResNet-18 hyperparameter sweep on BNCI2014001.

The current Raw_ResNet18 only exposes ``dropout`` to config; the rest of the
architecture (block widths, stem, etc.) is hard-coded. We sweep training and
regularization hyperparameters around it.
"""

BASE_CONFIG = "raw_bci_resnet18"
METRIC = "val_acc"
DIRECTION = "maximize"
MAX_EPOCHS = 80


def define_space(trial) -> dict:
    return {
        "training.lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "training.weight_decay": trial.suggest_float("wd", 1e-5, 0.3, log=True),
        "training.label_smoothing": trial.suggest_float("ls", 0.0, 0.3),
        "training.batch_size": trial.suggest_categorical("bs", [32, 64, 128]),
        "training.gradient_clip_norm": trial.suggest_float("gclip", 0.5, 5.0),
        "regularization.gaussian_std": trial.suggest_float("gnoise", 0.0, 0.3),
        "regularization.channel_dropout": trial.suggest_float("chdrop", 0.0, 0.4),
        "regularization.max_shift": trial.suggest_int("max_shift", 0, 30),
        "model.dropout": trial.suggest_float("dropout", 0.1, 0.7),
    }
