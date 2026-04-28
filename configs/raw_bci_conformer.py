from .base import base_config


def get_config():
    """Raw signal EEG Conformer configuration."""
    c = base_config()

    c.preprocessing.representation = "raw"
    c.data.dataset = "bnci2014001"
    c.eeg.num_classes = 4

    c.model.arch = "raw_conformer"
    c.model.checkpoint_name = "Raw_Conformer_v1"
    c.model.emb_size = 40
    c.model.depth = 6
    c.model.num_heads = 8
    c.model.ff_expansion = 4
    c.model.temporal_kernel = 25
    c.model.pool_kernel = 75
    c.model.pool_stride = 15
    c.model.dropout = 0.5
    c.model.cls_hidden = 256

    c.lock()
    return c