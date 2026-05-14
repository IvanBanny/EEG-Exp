"""EEGNet hyperparameter sweep on Our5Class."""

BASE_CONFIG = "raw_eegnet"
METRIC = "val_acc"
DIRECTION = "maximize"
MAX_EPOCHS = 80


def define_space(trial) -> dict:
    return {
        "training.lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "training.weight_decay": trial.suggest_float("wd", 1e-5, 0.3, log=True),
        "training.label_smoothing": trial.suggest_float("ls", 0.0, 0.3),
        "training.batch_size": trial.suggest_categorical("bs", [64, 128, 256]),
        "regularization.gaussian_std": trial.suggest_float("gnoise", 0.0, 0.3),
        "regularization.channel_dropout": trial.suggest_float("chdrop", 0.0, 0.4),
        "regularization.max_shift": trial.suggest_int("max_shift", 0, 30),
        "model.dropout": trial.suggest_float("dropout", 0.1, 0.7),
        "model.f1": trial.suggest_categorical("f1", [4, 8, 16, 32]),
        "model.d": trial.suggest_categorical("d", [1, 2, 4]),
        "model.f2": trial.suggest_categorical("f2", [8, 16, 32, 64]),
        "model.kernel_length": trial.suggest_categorical("kernel_length", [32, 64, 96, 128]),
        "model.sep_kernel": trial.suggest_categorical("sep_kernel", [8, 16, 32]),
    }
