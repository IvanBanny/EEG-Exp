"""EEG-Conformer hyperparameter sweep on BNCI2014001."""

BASE_CONFIG = "raw_bci_conformer"
METRIC = "val_acc"
DIRECTION = "maximize"
MAX_EPOCHS = 60


def define_space(trial) -> dict:
    # emb_size choices are all divisible by 4 and 8 so heads choice is unconstrained
    return {
        "training.lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "training.weight_decay": trial.suggest_float("wd", 1e-5, 0.3, log=True),
        "training.label_smoothing": trial.suggest_float("ls", 0.0, 0.3),
        "training.batch_size": trial.suggest_categorical("bs", [64, 128, 256]),
        "regularization.gaussian_std": trial.suggest_float("gnoise", 0.0, 0.3),
        "regularization.channel_dropout": trial.suggest_float("chdrop", 0.0, 0.4),
        "model.dropout": trial.suggest_float("dropout", 0.1, 0.6),
        "model.emb_size": trial.suggest_categorical("emb_size", [32, 40, 64, 80]),
        "model.depth": trial.suggest_int("depth", 2, 8),
        "model.num_heads": trial.suggest_categorical("num_heads", [4, 8]),
        "model.ff_expansion": trial.suggest_categorical("ff_expansion", [2, 4]),
        "model.temporal_kernel": trial.suggest_categorical("temporal_kernel", [15, 25, 35, 51]),
        "model.pool_kernel": trial.suggest_categorical("pool_kernel", [25, 50, 75, 100]),
        "model.pool_stride": trial.suggest_categorical("pool_stride", [5, 10, 15, 25]),
        "model.cls_hidden": trial.suggest_categorical("cls_hidden", [64, 128, 256]),
    }
