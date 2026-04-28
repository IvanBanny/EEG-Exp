from .base import base_config


def get_config():
    """STFT CNN-BiLSTM configuration."""
    c = base_config()

    c.data.dataset = "bnci2014001"
    c.eeg.num_classes = 4

    c.model.arch = "stft_cnn_bilstm"
    c.model.checkpoint_name = "STFT_CNN_BiLSTM_v1"
    c.model.hidden_dim = 64
    c.model.rnn_layers = 2
    c.model.dropout = 0.5
    c.model.se_reduction = 4

    c.lock()
    return c