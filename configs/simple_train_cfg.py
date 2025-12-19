import ml_collections

def get_config():
    """Returns the default configuration for the simple trainer script."""
    config = ml_collections.ConfigDict()

    # Top level
    config.seed = 4269

    # EEG / sensor setup
    config.eeg = ml_collections.ConfigDict()
    config.eeg.in_channels = 22
    config.eeg.num_classes = 5

    # Preprocessing
    config.preprocessing = ml_collections.ConfigDict()
    config.preprocessing.resample_rate = 250
    config.preprocessing.freq_fork = (4, 40)
    config.preprocessing.t_fork = (0, 6)
    config.preprocessing.window_sec = 3.0
    config.preprocessing.window_overlap = 0.9
    config.preprocessing.stft_nperseg = 64
    config.preprocessing.stft_overlap = 48
    config.preprocessing.use_cache = True
    config.preprocessing.cache_path = "./moabb_cache"

    # Regularization
    config.regularization = ml_collections.ConfigDict()
    config.regularization.clip_sigma = 4.0
    config.regularization.gaussian_std = 0.05
    config.regularization.scale_fork = (0.9, 1.1)
    config.regularization.max_shift = 10
    config.regularization.channel_dropout = 0.1

    # Loading
    config.data = ml_collections.ConfigDict()
    config.data.num_workers = 4

    # BiLSTM config
    config.model = ml_collections.ConfigDict()
    config.model.architecture = "CNN_BiLSTM"
    config.model.checkpoint_name = "CNN_BiLSTM_v1"

    config.model.cnnbilstm = ml_collections.ConfigDict()
    config.model.cnnbilstm.hidden_dim = 64
    config.model.cnnbilstm.rnn_layers = 2
    config.model.cnnbilstm.dropout = 0.5

    # Training loop
    config.training = ml_collections.ConfigDict()

    config.training.optimizer = "AdamW"
    config.training.loss_function = "CrossEntropyLoss"

    config.training.batch_size = 32
    config.training.epochs = 1000
    config.training.lr = 0.001
    config.training.weight_decay = 0.01
    config.training.gradient_clip_norm = 2.0

    # Early Stopping
    config.training.es_patience = 48
    config.training.es_min_delta = 0.0

    _patience_cycle = 16
    config.training.rollback_patience = _patience_cycle
    config.training.lr_patience = _patience_cycle

    config.training.rollback_min_delta = 0.0
    config.training.rollback_on_disk = False
    config.training.lr_factor = 0.5
    config.training.min_lr = 1e-7

    config.lock()
    return config
