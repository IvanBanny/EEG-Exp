"""A 1D CNN-BiLSTM implementation for raw EEG signal."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SE1DBlock(nn.Module):
    """Squeeze-and-Excitation block for 1D feature maps.

    Channel-wise attention over (B, C, T) tensors.
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Raw_CNN_BiLSTM(nn.Module):
    """1D CNN + (Bi)LSTM for raw EEG signal MI classification.

    Expects 3D input: (batch, eeg_channels, time_samples).
    Temporal convolutions extract local features, then an LSTM models temporal
    dependencies.

    The `causal` flag switches between the offline-benchmark default (BiLSTM
    with global-mean pooling) and the deployment-honest variant (unidirectional
    LSTM with last-hidden-state read-out). Causal mode classifies based on the
    end of the input window only - matching live streaming inference where the
    future is unknown.

    Args:
        in_channels: Number of EEG channels.
        num_classes: Number of MI classes.
        hidden_dim: Hidden dimension for CNN and RNN.
        rnn_layers: Number of LSTM layers.
        dropout: Dropout rate.
        pool_factor: Temporal downsampling factor before LSTM.
        se_reduction: SE block channel reduction ratio.
        causal: If True, use a unidirectional LSTM and read off only the last
            timestep. If False (default), use a BiLSTM with mean pooling.
    """
    def __init__(
            self,
            in_channels: int = 22,
            num_classes: int = 5,
            hidden_dim: int = 64,
            rnn_layers: int = 2,
            dropout: float = 0.5,
            pool_factor: int = 8,
            se_reduction: int = 4,
            causal: bool = False,
    ):
        super().__init__()
        self.causal = causal

        # Temporal feature extraction
        # kernel_size=25 at 250Hz ~ 100ms receptive field
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=25, padding=12)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=13, padding=6)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.se = SE1DBlock(hidden_dim, reduction=se_reduction)

        # Temporal downsampling before LSTM
        self.pool = nn.AvgPool1d(kernel_size=pool_factor, stride=pool_factor)

        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=not causal,
            dropout=dropout if rnn_layers > 1 else 0
        )

        # Classifier head: BiLSTM concatenates fwd+bwd, causal LSTM does not.
        head_dim = hidden_dim * (1 if causal else 2)
        self.fc = nn.Linear(head_dim, num_classes)

    @classmethod
    def from_config(cls, cfg):
        """Instantiate from an experiment ConfigDict."""
        return cls(
            in_channels=cfg.eeg.in_channels,
            num_classes=cfg.eeg.num_classes,
            hidden_dim=cfg.model.hidden_dim,
            rnn_layers=cfg.model.rnn_layers,
            dropout=cfg.model.dropout,
            pool_factor=cfg.model.pool_factor,
            se_reduction=cfg.model.se_reduction,
            causal=cfg.model.get("causal", False),
        )

    def forward(self, x):
        # x: (B, C, T)  e.g. (B, 22, 750)
        x = self.conv1(x)  # (B, H, T)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv2(x)  # (B, H, T)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.se(x)  # (B, H, T)

        x = self.pool(x)  # (B, H, T // pool_factor)
        x = x.permute(0, 2, 1)  # (B, T // pool_factor, H)

        x, _ = self.rnn(x)
        # Causal: classify on the last timestep only (deployment-honest).
        # BiLSTM: global mean pool (offline default).
        x = x[:, -1, :] if self.causal else x.mean(dim=1)

        logits = self.fc(x)  # (B, num_classes)
        return logits


if __name__ == "__main__":
    # 3 seconds at 250 Hz = 750 samples
    model = Raw_CNN_BiLSTM(
        in_channels=22,
        num_classes=5,
        hidden_dim=64,
        rnn_layers=2,
        dropout=0.5,
        pool_factor=8,
        se_reduction=4,
    )

    x = torch.randn(16, 22, 750)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")