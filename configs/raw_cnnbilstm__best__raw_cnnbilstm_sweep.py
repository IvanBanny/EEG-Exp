"""Best config from study 'raw_cnnbilstm_sweep' (base: raw_cnnbilstm)."""

from .raw_cnnbilstm import get_config as _base_get_config


def get_config():
    """Best params from sweep study 'raw_cnnbilstm_sweep'."""
    c = _base_get_config()
    with c.unlocked():
        c.training.lr = 0.00019392454177300671
        c.training.weight_decay = 0.030487098885797134
        c.training.label_smoothing = 0.23581570829269866
        c.training.batch_size = 64
        c.regularization.gaussian_std = 0.18632280484879613
        c.regularization.channel_dropout = 0.031048629147864076
        c.regularization.max_shift = 12
        c.model.dropout = 0.543796833835001
        c.model.hidden_dim = 64
        c.model.rnn_layers = 1
        c.model.pool_factor = 16
    c.lock()
    return c
