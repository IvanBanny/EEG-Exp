"""Dataset factory - build train/val datasets from config.

This module is the single canonical entry point for turning a config
into Torch datasets. It owns:

- Per-dataset MOABB plumbing (Our5Class, BNCI2014_001).
- Train/val splitting via `cfg.data.split_mode`.
- Subject selection via `cfg.data.subjects`.
- Cache routing that includes every preprocessing key affecting the
  cached arrays (`cache_tag` below). The default MOABB cache is keyed
  by paradigm hash only; changing `window_sec`, `window_overlap`,
  `t_fork`, or `freq_fork` can therefore silently reuse stale arrays.
  Including the tag in the cache directory makes that footgun
  impossible.
- Normalizer fitting (`per_window`, `per_subject`, `per_channel`).

Public surface:
    build_datasets(cfg)
    get_event_names(name)
    cache_tag(cfg)
"""

from pathlib import Path
from typing import Optional

import numpy as np
from moabb.paradigms import MotorImagery

from .torch_dataset import EEGDataset, EEGDatasetConfig, SubsetEEGDataset
from .transforms import PerSubjectZScore, PerChannelZScore
from ..data_proc import Our5Class

# Event lists per dataset.
_DATASET_EVENTS = {
    "our5class": (
        ["left_hand", "right_hand", "left_leg", "right_leg", "tongue"],
        5,
    ),
    "our4class": (
        ["left_hand", "right_hand", "left_leg", "right_leg"],
        4,
    ),
    "bnci2014001": (
        ["left_hand", "right_hand", "feet", "tongue"],
        4,
    ),
}

# Per-dataset legal `split_mode` values. First entry is the default.
_DATASET_SPLITS = {
    "our5class":   ("last_poly5",),
    "our4class":   ("last_poly5",),
    "bnci2014001": ("last_run", "session"),
}


def get_event_names(name):
    """Return the ordered list of class names for a dataset. Order matches
    the integer labels emitted by MOABB's MotorImagery paradigm."""
    if name not in _DATASET_EVENTS:
        raise ValueError(f"Unknown dataset '{name}'")
    return list(_DATASET_EVENTS[name][0])


def cache_tag(cfg) -> str:
    """Deterministic short tag for the MOABB array cache directory.

    Includes every preprocessing key that affects the cached arrays so
    that two configs that differ in any of them never collide on the
    same cache directory. `.` and `-` are mapped to filesystem-safe
    characters.

    Args:
        cfg: Experiment ConfigDict.

    Returns:
        Tag string, eg. `t1p0_4p0_w2p0_o0p95_f4p0_40p0_slast_run`.
    """
    a, b = cfg.preprocessing.t_fork
    f1, f2 = cfg.preprocessing.freq_fork
    split_mode = _resolve_split_mode(cfg)
    raw = (f"t{a}_{b}"
           f"_w{cfg.preprocessing.window_sec}"
           f"_o{cfg.preprocessing.window_overlap}"
           f"_f{f1}_{f2}"
           f"_s{split_mode}")
    return raw.replace(".", "p").replace("-", "m")


def _resolve_split_mode(cfg) -> str:
    """Read cfg.data.split_mode with the per-dataset default + validation."""
    name = cfg.data.dataset
    allowed = _DATASET_SPLITS[name]
    mode = cfg.data.get("split_mode", None) if hasattr(cfg.data, "get") else None
    # ml_collections ConfigDict supports .get but tuples don't; guard both.
    if mode is None:
        mode = allowed[0]
    if mode not in allowed:
        raise ValueError(
            f"Invalid cfg.data.split_mode='{mode}' for dataset '{name}'. "
            f"Allowed: {list(allowed)}."
        )
    return mode


def _cache_root(cfg) -> Path:
    """Resolve the full cache root for this cfg: `<base>/<dataset>/<repr>/<tag>`."""
    return (Path(cfg.preprocessing.cache_path)
            / cfg.data.dataset
            / cfg.preprocessing.representation
            / cache_tag(cfg))


def _paradigm(cfg, events, n_classes) -> MotorImagery:
    """Build the MOABB MotorImagery paradigm from cfg.

    IMPORTANT (BNCI2014_001 / BCI IV 2a):
    MOABB's MotorImagery paradigm applies its own anchor offset before
    epoching: `effective_tmin = self.tmin + dataset.interval[0]` (see
    moabb/paradigms/base.py:408). For BNCI2014_001 the dataset's
    `interval = [2, 6]`, so `cfg.preprocessing.t_fork` is measured
    FROM THE CUE ONSET, not from the trial start.

    Trial structure (seconds, trial-time):
        [0, 2)  fixation cross  (trial start = 0)
        [2, 3.25)  visual cue   (cue onset = 2)
        [3, 6]  motor imagery   (MI period)
        [6, 8.5]  post-trial rest

    So to crop the MI period only, use `t_fork = (1.0, 4.0)`. To crop
    the reference EEGEncoder paper's window (trial-time [1.5, 6.0]),
    use `t_fork = (-0.5, 4.0)`. Our5Class uses `interval = [0, 6]`, so
    its t_fork is trial-relative as the name suggests; only BCI 2a
    needs this adjustment.
    """
    return MotorImagery(
        events=events,
        n_classes=n_classes,
        fmin=cfg.preprocessing.freq_fork[0],
        fmax=cfg.preprocessing.freq_fork[1],
        resample=cfg.preprocessing.resample_rate,
        tmin=cfg.preprocessing.t_fork[0],
        tmax=cfg.preprocessing.t_fork[1],
    )


def _ds_common(cfg) -> dict:
    return dict(
        representation=cfg.preprocessing.representation,
        window_sec=cfg.preprocessing.window_sec,
        window_overlap=cfg.preprocessing.window_overlap,
        stft_nperseg=cfg.preprocessing.stft_nperseg,
        stft_overlap=cfg.preprocessing.stft_overlap,
        use_cache=cfg.preprocessing.use_cache,
        regen_cache=cfg.preprocessing.regen_cache,
        dtype=np.float32,
    )


def _subjects(cfg) -> Optional[list[int]]:
    """Read cfg.data.subjects if present, else None (= all subjects)."""
    if not hasattr(cfg.data, "get"):
        return None
    s = cfg.data.get("subjects", None)
    if not s:
        return None
    return [int(x) for x in s]


def _fit_per_subject_normalizer(train_ds) -> PerSubjectZScore:
    """Fit a PerSubjectZScore on a training dataset or subset.

    Groups windows by the subject column of the underlying metadata and
    computes scalar (mean, std) per subject. Stats are derived from the
    *training* split only; the same fitted normalizer is then applied to
    both train and val windows of that subject at sample-fetch time.
    """
    full_data, subjects = _resolve_train_arrays(train_ds)
    by_subject = {}
    for sid in np.unique(subjects):
        mask = subjects == sid
        by_subject[int(sid)] = full_data[mask]
    return PerSubjectZScore().fit(by_subject)


def _fit_per_channel_normalizer(train_ds) -> PerChannelZScore:
    """Fit a PerChannelZScore on a training dataset or subset.

    Per-(subject, channel) (mean, std) computed over all training
    windows of each subject. Matches the EEGEncoder paper's
    `StandardScaler`-per-channel preprocessing.
    """
    full_data, subjects = _resolve_train_arrays(train_ds)
    by_subject = {}
    for sid in np.unique(subjects):
        mask = subjects == sid
        by_subject[int(sid)] = full_data[mask]
    return PerChannelZScore().fit(by_subject)


def _resolve_train_arrays(train_ds):
    """Return (data_tensor, subject_ids_array) for a (Subset)EEGDataset."""
    if isinstance(train_ds, SubsetEEGDataset):
        idx = np.asarray(train_ds.indices)
        data = train_ds.parent.data[idx]
        subjects = train_ds.parent.metadata["subject"].to_numpy()[idx]
    else:
        data = train_ds.data
        subjects = train_ds.metadata["subject"].to_numpy()
    return data, subjects


def _maybe_fit_normalizer(cfg, train_ds, val_ds):
    """Return a fitted normalizer (or None) per cfg.preprocessing.normalize.

    Modes:
        "per_window": legacy single-window z-score, applied at sample
            fetch time by `ZScoreNormalize`. Returns None.
        "per_subject": scalar (mean, std) per subject fitted on the
            train split.
        "per_channel": per-(subject, channel) (mean, std) fitted on the
            train split.

    Also flips `return_subject_id=True` on the underlying EEGDataset(s)
    so the per-sample transform pipeline receives the subject id.
    """
    mode = getattr(cfg.preprocessing, "normalize", "per_window")
    if mode == "per_window":
        return None
    if mode == "per_subject":
        normalizer = _fit_per_subject_normalizer(train_ds)
    elif mode == "per_channel":
        normalizer = _fit_per_channel_normalizer(train_ds)
    else:
        raise ValueError(
            f"Unknown cfg.preprocessing.normalize='{mode}'. "
            f"Expected 'per_window', 'per_subject' or 'per_channel'."
        )

    for ds in (train_ds, val_ds):
        ds.return_subject_id = True
    n_subj = len(normalizer.stats)
    print(f"Fitted {type(normalizer).__name__} on {n_subj} subjects from train split.")
    return normalizer


def build_datasets(cfg):
    """Build train and val EEG datasets from config.

    Args:
        cfg: Experiment ConfigDict. Requires `cfg.data.dataset` to be
            set. Optional `cfg.data.subjects` (list[int]) limits the
            build to a subset of subjects. Optional
            `cfg.data.split_mode` selects the train/val split (legal
            values depend on the dataset).

    Returns:
        (train_dataset, val_dataset, normalizer) tuple. `normalizer` is
        a fitted PerSubjectZScore / PerChannelZScore when
        `cfg.preprocessing.normalize` is `"per_subject"` /
        `"per_channel"`, else None.
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

    # Validate split_mode early so we fail fast on configuration errors.
    _resolve_split_mode(cfg)

    paradigm = _paradigm(cfg, events, n_classes)
    common = _ds_common(cfg)

    if name in ("our5class", "our4class"):
        train_ds, val_ds = _build_our(cfg, name, paradigm, common)
    elif name == "bnci2014001":
        train_ds, val_ds = _build_bnci2014001(cfg, paradigm, common)
    else:
        raise ValueError(f"Unhandled dataset '{name}'")

    normalizer = _maybe_fit_normalizer(cfg, train_ds, val_ds)
    return train_ds, val_ds, normalizer


def _build_our(cfg, name, paradigm, common):
    """Our5Class/Our4Class: split by last poly5 file per subject. The
    MOABB paradigm filters to its `events` list, so the 4-class variant
    simply drops tongue trials. Cache path is keyed by `name` and the
    preprocessing tag so 4-class/5-class and different cropping
    configurations cannot collide."""
    moabb_train = Our5Class(
        data_path=cfg.data.data_path, get_last_trials_xor=False
    )
    moabb_val = Our5Class(
        data_path=cfg.data.data_path, get_last_trials_xor=True
    )

    subjects = _subjects(cfg)
    cache_root = _cache_root(cfg)
    train_config = EEGDatasetConfig(
        paradigm=paradigm,
        cache_path=cache_root / "train",
        subjects=subjects,
        **common,
    )
    val_config = EEGDatasetConfig(
        paradigm=paradigm,
        cache_path=cache_root / "val",
        subjects=subjects,
        **common,
    )

    train_ds = EEGDataset(moabb_train, train_config)
    val_ds = EEGDataset(moabb_val, val_config)
    return train_ds, val_ds


def _build_bnci2014001(cfg, paradigm, common):
    """BNCI2014_001 (BCI IV 2a). Two split modes:

    - `last_run` (default): val = last run per subject, train = the
      rest. The deployment-honest convention.
    - `session`: val = "1test" session, train = "0train" session.
      Matches the EEGEncoder paper protocol.
    """
    from moabb.datasets import BNCI2014_001

    subjects = _subjects(cfg)
    full_config = EEGDatasetConfig(
        paradigm=paradigm,
        cache_path=_cache_root(cfg),
        subjects=subjects,
        **common,
    )
    full_ds = EEGDataset(BNCI2014_001(), full_config)

    meta = full_ds.metadata
    split_mode = _resolve_split_mode(cfg)
    if split_mode == "last_run":
        last_run = meta.groupby("subject")["run"].transform("max")
        train_mask = (meta["run"] != last_run).to_numpy()
        val_mask = (meta["run"] == last_run).to_numpy()
    elif split_mode == "session":
        # MOABB BNCI2014_001 session ids are "0train" / "1test".
        session = meta["session"].astype(str)
        train_mask = session.str.contains("train").to_numpy()
        val_mask = session.str.contains("test").to_numpy()
        if not train_mask.any() or not val_mask.any():
            raise RuntimeError(
                f"Session split failed: train={train_mask.sum()} "
                f"val={val_mask.sum()}. Unique sessions: "
                f"{sorted(set(session.tolist()))}"
            )
    else:
        # _resolve_split_mode already validated this; defensive only.
        raise ValueError(f"Unhandled split_mode '{split_mode}'")

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    train_ds = SubsetEEGDataset(full_ds, train_idx)
    val_ds = SubsetEEGDataset(full_ds, val_idx)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds
