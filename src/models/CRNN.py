"""A CNN-BiLSTM implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_BiLSTM(nn.Module):
    """A simple CNN + BiLSTM implementation for MI classification.

    Args:
        in_channels: Number of EEG channels.
        freq_bins: Number of frequency bins in STFT.
        num_classes: Number of MI classes.
        hidden_dim: Hidden dimension for CNN and RNN.
        rnn_layers: Number or BiLSTM layers.
        dropout: Dropout rate.
    """
    def __init__(
            self,
            in_channels: int = 22,
            freq_bins: int = 40,
            num_classes: int = 5,
            hidden_dim: int = 64,
            rnn_layers: int = 2,
            dropout: float = 0.5
    ):
        super().__init__()

        # CNN: collapse channels and frequency into features
        self.conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=(freq_bins, 1))
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # BiLSTM: temporal modeling
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0
        )

        # Classifier
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: [B, C, F, T]
        x = self.conv(x)  # [B, H, 1, T]
        x = self.bn(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = x.squeeze(2).permute(0, 2, 1)  # [B, T, H]

        x, _ = self.rnn(x)  # [B, T, 2H]
        x = x.mean(dim=1)  # [B, 2H] - global average pooling

        logits = self.fc(x)  # [B, num_classes]
        return logits


if __name__ == "__main__":
    model = CNN_BiLSTM(
        in_channels=22,
        freq_bins=33,
        num_classes=5,
        hidden_dim=64,
        rnn_layers=2,
        dropout=0.5
    )

    x = torch.randn(16, 22, 33, 48)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
