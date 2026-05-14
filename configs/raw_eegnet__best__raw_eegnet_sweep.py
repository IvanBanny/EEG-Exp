"""Best config from study 'raw_eegnet_sweep' (base: raw_eegnet)."""

from .raw_eegnet import get_config as _base_get_config


def get_config():
    """Best params from sweep study 'raw_eegnet_sweep'."""
    c = _base_get_config()
    with c.unlocked():
        c.training.lr = 0.0010314883004894394
        c.training.weight_decay = 7.61257610716967e-05
        c.training.label_smoothing = 0.21291997401291063
        c.training.batch_size = 256
        c.regularization.gaussian_std = 0.1770958578069985
        c.regularization.channel_dropout = 0.17668761628633284
        c.regularization.max_shift = 5
        c.model.dropout = 0.2582561250056456
        c.model.f1 = 32
        c.model.d = 4
        c.model.f2 = 64
        c.model.kernel_length = 64
        c.model.sep_kernel = 32
    c.lock()
    return c
