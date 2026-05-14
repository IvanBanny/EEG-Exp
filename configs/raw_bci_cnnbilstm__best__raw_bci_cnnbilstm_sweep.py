"""Best config from study 'raw_bci_cnnbilstm_sweep' (base: raw_bci_cnnbilstm)."""

from .raw_bci_cnnbilstm import get_config as _base_get_config


def get_config():
    """Best params from sweep study 'raw_bci_cnnbilstm_sweep'."""
    c = _base_get_config()
    with c.unlocked():
        c.training.lr = 0.00012017557575394685
        c.training.weight_decay = 0.023811040217340144
        c.training.label_smoothing = 0.1243193963236433
        c.training.batch_size = 64
        c.regularization.gaussian_std = 0.18053465273362967
        c.regularization.channel_dropout = 0.00577215301721911
        c.model.dropout = 0.1099426396137242
        c.model.hidden_dim = 32
        c.model.rnn_layers = 1
    c.lock()
    return c
