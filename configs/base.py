"""Base configuration shared across all experiments.

Experiment configs should call base_config(), override what they need,
add model-specific params, then lock().
"""

import ml_collections


def base_config():
    """Return unlocked base ConfigDict with all shared defaults."""
    c = ml_collections.ConfigDict()

    c.seed = 4269

    # EEG / sensor setup
    c.eeg = ml_collections.ConfigDict()
    c.eeg.in_channels = 22
    c.eeg.num_classes = 5

    # Preprocessing
    c.preprocessing = ml_collections.ConfigDict()
    c.preprocessing.representation = "stft"  # "stft" or "raw"
    c.preprocessing.resample_rate = 250
    c.preprocessing.freq_fork = (4, 40)
    c.preprocessing.t_fork = (0, 6)
    c.preprocessing.window_sec = 3.0
    c.preprocessing.window_overlap = 0.9
    c.preprocessing.stft_nperseg = 64  # ignored when representation="raw"
    c.preprocessing.stft_overlap = 48  # ignored when representation="raw"
    c.preprocessing.use_cache = True
    c.preprocessing.regen_cache = False  # force-recompute cached arrays on next run
    c.preprocessing.cache_path = "./moabb_cache"

    # Regularization
    c.regularization = ml_collections.ConfigDict()
    c.regularization.clip_sigma = 4.0
    c.regularization.gaussian_std = 0.15
    c.regularization.scale_fork = (0.8, 1.2)
    c.regularization.max_shift = 10
    c.regularization.channel_dropout = 0.2

    # Data
    c.data = ml_collections.ConfigDict()
    c.data.dataset = "our5class"
    c.data.data_path = "our_data/our_5_class"  # only used for our5class
    c.data.num_workers = 8

    # Model - arch-specific params added by experiment configs
    c.model = ml_collections.ConfigDict()
    c.model.arch = ""  # set by experiment config
    c.model.checkpoint_name = ""  # set by experiment config

    # Training
    c.training = ml_collections.ConfigDict()
    c.training.optimizer = "AdamW"
    c.training.loss_function = "CrossEntropyLoss"
    c.training.label_smoothing = 0.2
    c.training.batch_size = 64
    c.training.epochs = 1000
    c.training.lr = 0.0003
    c.training.weight_decay = 0.1
    c.training.gradient_clip_norm = 2.0

    # Early stopping
    c.training.es_patience = 48
    c.training.es_min_delta = 0.0

    _patience_cycle = 16
    c.training.rollback_patience = _patience_cycle
    c.training.lr_patience = _patience_cycle

    c.training.rollback_min_delta = 0.0
    c.training.rollback_on_disk = False
    c.training.lr_factor = 0.5
    c.training.min_lr = 1e-7

    return c