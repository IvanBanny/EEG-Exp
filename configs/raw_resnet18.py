from .base import base_config


def get_config():
    """Raw signal ResNet-18 with SE blocks configuration."""
    c = base_config()

    c.preprocessing.representation = "raw"

    c.model.arch = "raw_resnet18"
    c.model.checkpoint_name = "Raw_ResNet18_v1"
    c.model.dropout = 0.5

    c.lock()
    return c