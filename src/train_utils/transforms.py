"""Canonical per-window transform pipeline for raw EEG.

Both `train.py` (pooled) and `train_subjects.py` (per-subject) build
their train/val pipelines through `build_transforms`. The function is
keyed on `cfg.preprocessing.representation` and on the augmentation
knobs in `cfg.regularization`: zero-knob augmentations are dropped
from the train pipeline (not just disabled) so they consume no RNG
state and so the per-subject runs are reproducible across reruns of
the same seed.
"""

from typing import Optional, Tuple

from ..loaders import (
    ChannelDropout,
    ClipOutliers,
    Compose,
    GaussianNoise,
    LogCompress,
    RandomScale,
    TimeShift,
    ZScoreNormalize,
)


def build_transforms(cfg, subject_normalizer=None) -> Tuple[Compose, Compose]:
    """Return `(train_transform, val_transform)` for the given config.

    Args:
        cfg: Experiment ConfigDict with `preprocessing.representation`
            and `regularization.{clip_sigma, gaussian_std, scale_fork,
            max_shift, channel_dropout}` populated.
        subject_normalizer: Fitted per-subject normalizer applied by
            `TransformWrapper`. When provided, the per-window
            `ZScoreNormalize` is omitted from the base pipeline to
            avoid double-normalization.

    Returns:
        `(train_transform, val_transform)`. Train applies the base
        pipeline plus any non-zero augmentations; val applies the base
        pipeline only.
    """
    base = [ClipOutliers(sigma=cfg.regularization.clip_sigma)]
    if cfg.preprocessing.representation == "stft":
        base.append(LogCompress())
    if subject_normalizer is None:
        base.append(ZScoreNormalize())

    aug = []
    if cfg.regularization.gaussian_std > 0:
        aug.append(GaussianNoise(std=cfg.regularization.gaussian_std))
    s_lo, s_hi = cfg.regularization.scale_fork
    if (s_lo, s_hi) != (1.0, 1.0):
        aug.append(RandomScale(scale_fork=cfg.regularization.scale_fork))
    if cfg.regularization.max_shift > 0:
        aug.append(TimeShift(max_shift=cfg.regularization.max_shift))
    if cfg.regularization.channel_dropout > 0:
        aug.append(ChannelDropout(p=cfg.regularization.channel_dropout))

    return Compose(base + aug), Compose(list(base))
