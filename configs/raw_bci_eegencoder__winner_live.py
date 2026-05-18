"""BCI IV 2a, EEGEncoder, 2 s MI-only windows, 200 ep + ES schedule.

Per-subject reference (9 subjects x 3 seeds, t_fork=(1.0, 4.0) after
MOABB's cue-anchor offset, 95 % overlap, per-subject scalar z-score,
no augmentation):

    win_acc 0.661 +- 0.159   win_kappa 0.548 +- 0.212
    trial_acc 0.682 +- 0.157  trial_kappa 0.576 +- 0.210

Adding `--max-epochs 500 --no-early-stop` to `train_subjects.py` lifts
this cell to trial_kappa 0.604 +- 0.223. `train.py` runs pooled - the
pooled numbers differ from the per-subject reference above.
"""

from .base import base_config


def get_config():
    c = base_config()
    with c.unlocked():
        c.seed = 4269

        c.eeg.in_channels = 22
        c.eeg.num_classes = 4

        c.preprocessing.representation = "raw"
        c.preprocessing.resample_rate = 250
        # BNCI2014_001 has interval[0]=2 so t_fork is cue-relative; the
        # MI-only window in trial-time [3, 6] s maps to (1.0, 4.0).
        c.preprocessing.t_fork = (1.0, 4.0)
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

        c.data.dataset = "bnci2014001"
        c.data.split_mode = "last_run"
        c.data.num_workers = 4

        c.model.arch = "eegencoder"
        c.model.checkpoint_name = "raw_bci_eegencoder__winner_live"
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
