"""An EEG-Conformer implementation.

Inspired by https://github.com/eeyhsong/EEG-Conformer/tree/main.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Raw_Conformer(nn.Module):
    """CNN + Transformer Encoder + FC for raw EEG signal classification.

    Args:
        in_channels: number of EEG channels.
        num_classes: number of MI classes.
        dropout: Dropout rate.

    """
    pass
