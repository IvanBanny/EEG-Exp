from .base import base_config


def get_config():
    """Raw signal CNN-BiLSTM configuration."""
    c = base_config()

    c.preprocessing.representation = "raw"
    c.preprocessing.regen_cache = False
    c.preprocessing.t_fork = (-2, 4)

    c.model.arch = "raw_cnn_bilstm"
    c.model.checkpoint_name = "Raw_CNN_BiLSTM"
    c.model.hidden_dim = 64
    c.model.rnn_layers = 2
    c.model.dropout = 0.5
    c.model.pool_factor = 8
    c.model.se_reduction = 4

    c.lock()
    return c
