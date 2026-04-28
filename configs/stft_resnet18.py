from .base import base_config


def get_config():
    """STFT ResNet-18 with SE blocks configuration."""
    c = base_config()

    c.model.arch = "stft_resnet18"
    c.model.checkpoint_name = "STFT_ResNet18_v1"
    c.model.dropout = 0.5

    c.lock()
    return c