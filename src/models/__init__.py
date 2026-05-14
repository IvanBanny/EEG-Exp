import torch
import torch.nn as nn
import torch.optim as optim

from .CRNN import STFT_CNN_BiLSTM
from .RawCRNN import Raw_CNN_BiLSTM
from .ResNet18 import STFT_ResNet18
from .RawResNet18 import Raw_ResNet18
from .RawConformer import Raw_Conformer
from .EEGNet import EEGNet

_MODEL_REGISTRY = {
    "stft_cnn_bilstm": STFT_CNN_BiLSTM,
    "raw_cnn_bilstm": Raw_CNN_BiLSTM,
    "stft_resnet18": STFT_ResNet18,
    "raw_resnet18": Raw_ResNet18,
    "raw_conformer": Raw_Conformer,
    "eegnet": EEGNet,
}


def build_model(cfg, device):
    """Instantiate model, criterion, optimizer, and scheduler from config.

    Args:
        cfg: Experiment ConfigDict with model.arch set.
        device: Torch device.

    Returns:
        (model, criterion, optimizer, scheduler) tuple.
    """
    arch = cfg.model.arch
    if arch not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown arch '{arch}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )

    model_cls = _MODEL_REGISTRY[arch]
    model = model_cls.from_config(cfg).to(device, dtype=torch.float32)

    criterion = getattr(nn, cfg.training.loss_function)(
        label_smoothing=cfg.training.label_smoothing
    )
    optimizer = getattr(optim, cfg.training.optimizer)(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=cfg.training.lr_patience,
        factor=cfg.training.lr_factor,
    )

    return model, criterion, optimizer, scheduler


__all__ = [
    "STFT_CNN_BiLSTM",
    "Raw_CNN_BiLSTM",
    "STFT_ResNet18",
    "Raw_ResNet18",
    "build_model",
]