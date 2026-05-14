"""Best config from study 'raw_bci_eegnet_sweep' (base: raw_bci_eegnet)."""

from .raw_bci_eegnet import get_config as _base_get_config


def get_config():
    """Best params from sweep study 'raw_bci_eegnet_sweep'."""
    c = _base_get_config()
    with c.unlocked():
        c.training.lr = 0.0008636748643128514
        c.training.weight_decay = 0.03341378064659032
        c.training.label_smoothing = 0.09564359773579063
        c.training.batch_size = 256
        c.regularization.gaussian_std = 0.08388081191306503
        c.regularization.channel_dropout = 0.015191057377716515
        c.regularization.max_shift = 20
        c.model.dropout = 0.3775380918751253
        c.model.f1 = 32
        c.model.d = 4
        c.model.f2 = 32
        c.model.kernel_length = 128
        c.model.sep_kernel = 8
    c.lock()
    return c
