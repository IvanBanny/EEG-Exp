"""Our5Class, EEGEncoder, 2 s windows, 200 ep + ES schedule.

Per-subject reference (14 subjects x 3 seeds, t_fork=(0, 6), 95 %
overlap, per-subject scalar z-score, no augmentation):

    win_acc 0.326 +- 0.099   win_kappa 0.158 +- 0.123
    trial_acc 0.381 +- 0.138  trial_kappa 0.226 +- 0.172

`train.py` runs the same hyperparameters pooled - one model on all 14
subjects' train data joined, validated on the joined held-out poly5
split. Pooled numbers differ from the per-subject reference above and
are not directly comparable to the within-subject literature.
"""

from .base import base_config


def get_config():
    c = base_config()
    with c.unlocked():
        c.seed = 4269

        c.eeg.in_channels = 22
        c.eeg.num_classes = 5

        c.preprocessing.representation = "raw"
        c.preprocessing.resample_rate = 250
        c.preprocessing.t_fork = (0.0, 6.0)
        c.preprocessing.window_sec = 2.0
        c.preprocessing.window_overlap = 0.95
        c.preprocessing.freq_fork = (4.0, 40.0)
        c.preprocessing.normalize = "per_subject"

        c.regularization.clip_sigma = 5.0
        c.regularization.gaussian_std = 0.0
        c.regularization.scale_fork = (1.0, 1.0)
        c.regularization.max_shift = 0
        c.regularization.channel_dropout = 0.0
        c.regularization.mixup_alpha = 0.0

        c.data.dataset = "our5class"
        c.data.split_mode = "last_poly5"
        c.data.data_path = "our_data/our_5_class"
        c.data.num_workers = 4

        c.model.arch = "eegencoder"
        c.model.checkpoint_name = "raw_our5_eegencoder__winner_live"
        c.model.f1 = 16
        c.model.d = 2
        c.model.kernel_length = 64
        c.model.sep_kernel = 16
        c.model.pool1 = 8
        c.model.pool2 = 7
        c.model.projector_dropout = 0.3
        c.model.n_branches = 5
        c.model.num_heads = 2
        c.model.num_layers = 2
        c.model.ffn_dim = 32
        c.model.tcn_depth = 2
        c.model.tcn_kernel = 4
        c.model.tcn_filters = 32
        c.model.tcn_dropout = 0.3
        c.model.branch_dropout = 0.3
        c.model.trm_dropout = 0.3

        c.training.optimizer = "AdamW"
        c.training.loss_function = "CrossEntropyLoss"
        c.training.label_smoothing = 0.1
        c.training.batch_size = 64
        c.training.epochs = 200
        c.training.num_runs = 1
        c.training.lr = 1e-3
        c.training.weight_decay = 0.0
        c.training.gradient_clip_norm = 2.0
        c.training.es_patience = 32
        c.training.rollback_patience = 16
        c.training.lr_patience = 16
        c.training.lr_factor = 0.5
        c.training.min_lr = 1e-7
    c.lock()
    return c
