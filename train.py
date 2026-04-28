"""Universal training script.

Usage:
    python train.py --config stft_cnnbilstm
    python train.py --config raw_cnnbilstm
    python train.py --config stft_resnet18
"""

import argparse
import importlib

import numpy as np
import torch

from src.loaders import (
    build_datasets,
    Compose, ClipOutliers, LogCompress, ZScoreNormalize,
    GaussianNoise, RandomScale, TimeShift, ChannelDropout,
)
from src.models import build_model
from src.train_utils import ClassicTrainer


def main():
    parser = argparse.ArgumentParser(description="Train an EEG-MI model.")
    parser.add_argument(
        "--config", required=True,
        help="Config module name in configs/ (e.g. stft_cnnbilstm)",
    )
    args = parser.parse_args()

    # Load config
    cfg_module = importlib.import_module(f"configs.{args.config}")
    cfg = cfg_module.get_config()

    # Seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_ds, val_ds = build_datasets(cfg)

    # Transforms - LogCompress only for STFT (compresses magnitude dynamic range)
    _base = [ClipOutliers(sigma=cfg.regularization.clip_sigma)]
    if cfg.preprocessing.representation == "stft":
        _base.append(LogCompress())
    _base.append(ZScoreNormalize())
    train_transform = Compose(_base + [
        GaussianNoise(std=cfg.regularization.gaussian_std),
        RandomScale(scale_fork=cfg.regularization.scale_fork),
        TimeShift(max_shift=cfg.regularization.max_shift),
        ChannelDropout(p=cfg.regularization.channel_dropout),
    ])
    val_transform = Compose(list(_base))

    # Model
    model, criterion, optimizer, scheduler = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{cfg.model.arch} | {n_params:,} params | {device}\n")

    # Train
    trainer = ClassicTrainer(
        model, criterion, optimizer,
        train_transform, val_transform,
        scheduler, device, cfg,
    )
    trainer.train_loop(train_set=train_ds, val_set=val_ds)


if __name__ == "__main__":
    main()