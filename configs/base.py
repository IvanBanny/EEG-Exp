"""Base config shared across all experiments.

Experiment configs call `base_config()`, override what they need, add
model-specific params, then `lock()`.
"""

import ml_collections


def base_config():
    """Return an unlocked base ConfigDict with the shared defaults."""
    c = ml_collections.ConfigDict()

    c.seed = 4269

    # EEG / sensor setup
    c.eeg = ml_collections.ConfigDict()
    c.eeg.in_channels = 22
    c.eeg.num_classes = 5

    # Preprocessing
    c.preprocessing = ml_collections.ConfigDict()
    c.preprocessing.representation = "raw"  # "raw" or "stft"
    c.preprocessing.resample_rate = 250
    c.preprocessing.freq_fork = (4, 40)
    # MOABB shifts MotorImagery(tmin, tmax) by `dataset.interval[0]`
    # before epoching, so t_fork is measured from the dataset's natural
    # anchor: trial start for Our5Class (interval[0]=0), cue onset for
    # BNCI2014_001 (interval[0]=2). See src/loaders/factory.py:_paradigm.
    # For BCI 2a MI-only, use t_fork=(1.0, 4.0).
    c.preprocessing.t_fork = (0, 6)
    c.preprocessing.window_sec = 2.0
    c.preprocessing.window_overlap = 0.95
    c.preprocessing.stft_nperseg = 64  # ignored when representation="raw"
    c.preprocessing.stft_overlap = 48  # ignored when representation="raw"
    # "per_window": single-window z-score (transforms.ZScoreNormalize).
    # "per_subject": scalar (mean, std) per subject, fit on train, reused
    #   for both train and val of that subject. Default.
    # "per_channel": per-(subject, channel) stats; matches the EEGEncoder
    #   paper convention.
    c.preprocessing.normalize = "per_subject"
    c.preprocessing.use_cache = True
    c.preprocessing.regen_cache = False  # force-recompute cached arrays
    c.preprocessing.cache_path = "./moabb_cache"

    # Regularization. Augmentation is off by default; non-zero knobs opt
    # the corresponding transform back into the train pipeline via
    # src/train_utils/transforms.build_transforms.
    c.regularization = ml_collections.ConfigDict()
    c.regularization.clip_sigma = 5.0
    c.regularization.gaussian_std = 0.0
    c.regularization.scale_fork = (1.0, 1.0)
    c.regularization.max_shift = 0
    c.regularization.channel_dropout = 0.0
    # Within-(subject, class) mixup. 0.0 disables; positive values draw
    # lambda ~ Beta(alpha, alpha) and mix paired samples at the dataset
    # level. See src/loaders/augmentation.IntraSubjectMixup.
    c.regularization.mixup_alpha = 0.0

    # Data
    c.data = ml_collections.ConfigDict()
    c.data.dataset = "our5class"
    c.data.data_path = "our_data/our_5_class"  # only used for our5class
    c.data.num_workers = 8

    # Model - arch-specific params added by experiment configs.
    c.model = ml_collections.ConfigDict()
    c.model.arch = ""
    c.model.checkpoint_name = ""
    # Optional end-of-window pool (last-K avg). When set, time-aware
    # models replace their terminal flatten / global mean with an
    # average over the most recent `endpool_ms` of input.
    c.model.endpool_ms = None

    # Training. Defaults match the headline winner schedule
    # (200 ep + ES patience 32, AdamW lr=1e-3, wd=0, label smoothing 0.1).
    c.training = ml_collections.ConfigDict()
    c.training.optimizer = "AdamW"
    c.training.loss_function = "CrossEntropyLoss"
    c.training.label_smoothing = 0.1
    c.training.batch_size = 64
    c.training.epochs = 200
    # Multi-seed averaging in pooled mode (train.py): >1 trains the same
    # arch num_runs times with seeds {seed, seed+1, ...}, keeps each
    # seed's TB run, and writes a separate `_agg/` run with mean / std /
    # n_active curves. Per-subject training (train_subjects.py) takes
    # seeds via CLI and ignores this field.
    c.training.num_runs = 1
    c.training.lr = 1e-3
    c.training.weight_decay = 0.0
    c.training.gradient_clip_norm = 2.0

    # Early stopping / rollback / LR plateau (32 / 16 / 16 cycle).
    c.training.es_patience = 32
    c.training.es_min_delta = 0.0

    _patience_cycle = 16
    c.training.rollback_patience = _patience_cycle
    c.training.lr_patience = _patience_cycle

    c.training.rollback_min_delta = 0.0
    c.training.rollback_on_disk = False
    c.training.lr_factor = 0.5
    c.training.min_lr = 1e-7

    return c