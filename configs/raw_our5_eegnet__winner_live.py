"""Our5Class, EEGNet, 1 s windows, 200 ep + ES schedule.

Per-subject reference (14 subjects x 3 seeds, t_fork=(0, 6), 95 %
overlap, per-subject scalar z-score, no augmentation):

    win_acc 0.290 +- 0.062   win_kappa 0.112 +- 0.078
    trial_acc 0.354 +- 0.113  trial_kappa 0.192 +- 0.142

`train.py` runs the same hyperparameters pooled; pooled numbers
differ from the per-subject reference above.
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
        c.preprocessing.window_sec = 1.0
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

        c.model.arch = "eegnet"
        c.model.checkpoint_name = "raw_our5_eegnet__winner_live"
        c.model.f1 = 8
        c.model.d = 2
        c.model.f2 = 16
        c.model.kernel_length = 64
        c.model.pool1 = 4
        c.model.pool2 = 8
        c.model.sep_kernel = 16
        c.model.dropout = 0.5
        c.model.depth_max_norm = 1.0
        c.model.cls_max_norm = 0.25
        c.model.endpool_ms = 500.0

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
