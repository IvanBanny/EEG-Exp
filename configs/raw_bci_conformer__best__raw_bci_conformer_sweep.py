"""Best config from study 'raw_bci_conformer_sweep' (base: raw_bci_conformer)."""

from .raw_bci_conformer import get_config as _base_get_config


def get_config():
    """Best params from sweep study 'raw_bci_conformer_sweep'."""
    c = _base_get_config()
    with c.unlocked():
        c.training.lr = 0.00018964397393021276
        c.training.weight_decay = 0.0024446597969694367
        c.training.label_smoothing = 0.06144472982951735
        c.training.batch_size = 64
        c.regularization.gaussian_std = 0.21490337536947318
        c.regularization.channel_dropout = 0.20709209895026987
        c.model.dropout = 0.15133503152864877
        c.model.emb_size = 64
        c.model.depth = 3
        c.model.num_heads = 4
        c.model.ff_expansion = 4
        c.model.temporal_kernel = 25
        c.model.pool_kernel = 100
        c.model.pool_stride = 25
        c.model.cls_hidden = 64
    c.lock()
    return c
