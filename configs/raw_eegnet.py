from .base import base_config


def get_config():
    """Raw signal EEGNet configuration on Our5Class."""
    c = base_config()

    c.preprocessing.representation = "raw"
    c.preprocessing.t_fork = (-2, 4)  # match raw_cnnbilstm so caches align
    c.preprocessing.regen_cache = True  # invalidate any stale cache from prior t_fork choices

    c.model.arch = "eegnet"
    c.model.checkpoint_name = "EEGNet_our5class_v1"
    c.model.f1 = 8
    c.model.d = 2
    c.model.f2 = 16
    c.model.kernel_length = 64
    c.model.pool1 = 4
    c.model.pool2 = 8
    c.model.sep_kernel = 16
    c.model.dropout = 0.5

    c.lock()
    return c
