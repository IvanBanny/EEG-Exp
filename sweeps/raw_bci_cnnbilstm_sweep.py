"""Hyperparameter sweep for raw_bci_cnnbilstm."""

BASE_CONFIG = "raw_bci_cnnbilstm"
METRIC = "val_acc"
DIRECTION = "maximize"
MAX_EPOCHS = 120


def define_space(trial) -> dict:
    """Return dotted-path overrides for the base config."""
    return {
        "training.lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "training.weight_decay": trial.suggest_float("wd", 1e-4, 0.3, log=True),
        "training.label_smoothing": trial.suggest_float("ls", 0.0, 0.3),
        "training.batch_size": trial.suggest_categorical("bs", [64, 128, 256, 512]),
        "regularization.gaussian_std": trial.suggest_float("gnoise", 0.0, 0.3),
        "regularization.channel_dropout": trial.suggest_float("chdrop", 0.0, 0.4),
        "model.dropout": trial.suggest_float("dropout", 0.1, 0.6),
        "model.hidden_dim": trial.suggest_categorical("hidden", [32, 64, 128]),
        "model.rnn_layers": trial.suggest_int("rnn_layers", 1, 3),
    }
