"""Raw CNN-BiLSTM hyperparameter sweep on Our5Class.

Starts from the winner config and overrides the HPs in `define_space`.
Run via `python sweep.py --sweep raw_cnnbilstm_sweep --trials 100`.
"""

BASE_CONFIG = "raw_our5_cnnbilstm__winner_live"
METRIC = "val_acc"
DIRECTION = "maximize"
MAX_EPOCHS = 80


def define_space(trial) -> dict:
    return {
        "training.lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "training.weight_decay": trial.suggest_float("wd", 1e-4, 0.3, log=True),
        "training.label_smoothing": trial.suggest_float("ls", 0.0, 0.3),
        "training.batch_size": trial.suggest_categorical("bs", [64, 128, 256, 512]),
        "regularization.gaussian_std": trial.suggest_float("gnoise", 0.0, 0.3),
        "regularization.channel_dropout": trial.suggest_float("chdrop", 0.0, 0.4),
        "regularization.max_shift": trial.suggest_int("max_shift", 0, 30),
        "model.dropout": trial.suggest_float("dropout", 0.1, 0.6),
        "model.hidden_dim": trial.suggest_categorical("hidden", [32, 64, 128]),
        "model.rnn_layers": trial.suggest_int("rnn_layers", 1, 3),
        "model.pool_factor": trial.suggest_categorical("pool_factor", [4, 8, 16]),
    }
