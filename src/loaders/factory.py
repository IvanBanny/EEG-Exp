"""Dataset factory - build train/val datasets from config."""

from pathlib import Path

import numpy as np
from moabb.paradigms import MotorImagery

from .torch_dataset import EEGDataset, EEGDatasetConfig, SubsetEEGDataset
from ..data_proc import Our5Class

# Event lists per dataset
_DATASET_EVENTS = {
    "our5class": (
        ["left_hand", "right_hand", "left_leg", "right_leg", "tongue"],
        5,
    ),
    "bnci2014001": (
        ["left_hand", "right_hand", "feet", "tongue"],
        4,
    ),
}


def build_datasets(cfg):
    """Build train and val EEG datasets from config.

    Args:
        cfg: Experiment ConfigDict. Requires cfg.data.dataset to be set.

    Returns:
        (train_dataset, val_dataset) tuple.
    """
    name = cfg.data.dataset
    if name not in _DATASET_EVENTS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(_DATASET_EVENTS.keys())}"
        )

    events, n_classes = _DATASET_EVENTS[name]
    if cfg.eeg.num_classes != n_classes:
        raise ValueError(
            f"cfg.eeg.num_classes={cfg.eeg.num_classes} but "
            f"dataset '{name}' has {n_classes} classes"
        )

    paradigm = MotorImagery(
        events=events,
        n_classes=n_classes,
        fmin=cfg.preprocessing.freq_fork[0],
        fmax=cfg.preprocessing.freq_fork[1],
        resample=cfg.preprocessing.resample_rate,
        tmin=cfg.preprocessing.t_fork[0],
        tmax=cfg.preprocessing.t_fork[1],
    )

    ds_common = dict(
        representation=cfg.preprocessing.representation,
        window_sec=cfg.preprocessing.window_sec,
        window_overlap=cfg.preprocessing.window_overlap,
        stft_nperseg=cfg.preprocessing.stft_nperseg,
        stft_overlap=cfg.preprocessing.stft_overlap,
        use_cache=cfg.preprocessing.use_cache,
        regen_cache=cfg.preprocessing.regen_cache,
        dtype=np.float32,
    )

    if name == "our5class":
        return _build_our5class(cfg, paradigm, ds_common)
    elif name == "bnci2014001":
        return _build_bnci2014001(cfg, paradigm, ds_common)


def _build_our5class(cfg, paradigm, ds_common):
    """Our5Class: split by last trial per subject."""
    moabb_train = Our5Class(
        data_path=cfg.data.data_path, get_last_trials_xor=False
    )
    moabb_val = Our5Class(
        data_path=cfg.data.data_path, get_last_trials_xor=True
    )

    cache_root = Path(cfg.preprocessing.cache_path) / "our5class" / cfg.preprocessing.representation
    train_config = EEGDatasetConfig(
        paradigm=paradigm,
        cache_path=cache_root / "train",
        **ds_common,
    )
    val_config = EEGDatasetConfig(
        paradigm=paradigm,
        cache_path=cache_root / "val",
        **ds_common,
    )

    train_ds = EEGDataset(moabb_train, train_config)
    val_ds = EEGDataset(moabb_val, val_config)
    return train_ds, val_ds


def _build_bnci2014001(cfg, paradigm, ds_common):
    """BNCI2014001 (BCI IV 2a): split by last run per subject."""
    from moabb.datasets import BNCI2014_001

    moabb_dataset = BNCI2014_001()
    full_config = EEGDatasetConfig(
        paradigm=paradigm,
        cache_path=Path(cfg.preprocessing.cache_path) / "bnci2014001" / cfg.preprocessing.representation,
        **ds_common,
    )

    full_ds = EEGDataset(moabb_dataset, full_config)

    # Val = last run per subject, train = everything else
    meta = full_ds.metadata
    last_run = meta.groupby("subject")["run"].transform("max")
    train_idx = np.where((meta["run"] != last_run).values)[0]
    val_idx = np.where((meta["run"] == last_run).values)[0]

    train_ds = SubsetEEGDataset(full_ds, train_idx)
    val_ds = SubsetEEGDataset(full_ds, val_idx)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds