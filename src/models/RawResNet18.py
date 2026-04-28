"""A 1D ResNet-18 with SE blocks for raw EEG signal."""

import torch
from torch import nn


class SE1DBlock(nn.Module):
    """Squeeze-and-Excitation block for 1D feature maps.

    Channel-wise attention over (B, C, T) tensors.
    """
    def __init__(self, in_channels, reduction=16):
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


class ResidualBlock1D(nn.Module):
    """1D residual block with optional SE attention."""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_se=True):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = downsample
        self.use_se = use_se
        if self.use_se:
            self.se = SE1DBlock(out_channels)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        out += identity
        out = self.silu(out)
        return out


class Raw_ResNet18(nn.Module):
    """1D ResNet-18 with SE blocks for raw EEG signal.

    Expects 3D input: (batch, eeg_channels, time_samples).
    Standard ResNet-18 structure adapted to 1D temporal convolutions.

    Args:
        num_classes: Number of MI classes.
        in_channels: Number of EEG channels.
        dropout_rate: Dropout before final FC.
    """
    def __init__(self, num_classes, in_channels=22, dropout_rate=0.5):
        super().__init__()

        # Stem: expand channels + 4x temporal downsample
        # 750 -> 375 (conv stride 2) -> 188 (maxpool stride 2)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.in_channels_idx = 64

        # ResNet-18: [2, 2, 2, 2] blocks
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

        self._init_weights()

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels_idx != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels_idx, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(ResidualBlock1D(self.in_channels_idx, out_channels, stride, downsample))
        self.in_channels_idx = out_channels

        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", a=0.01, nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @classmethod
    def from_config(cls, cfg):
        """Instantiate from an experiment ConfigDict."""
        return cls(
            num_classes=cfg.eeg.num_classes,
            in_channels=cfg.eeg.in_channels,
            dropout_rate=cfg.model.dropout,
        )

    def forward(self, x):
        # x: (B, C, T) e.g. (B, 22, 750)
        x = self.stem(x)       # (B, 64, 188)

        x = self.layer1(x)     # (B, 64, 188)
        x = self.layer2(x)     # (B, 128, 94)
        x = self.layer3(x)     # (B, 256, 47)
        x = self.layer4(x)     # (B, 512, 24)

        x = self.avgpool(x)    # (B, 512, 1)
        x = torch.flatten(x, 1)  # (B, 512)
        x = self.classifier(x)   # (B, num_classes)

        return x


if __name__ == "__main__":
    # 3 seconds at 250 Hz = 750 samples
    model = Raw_ResNet18(num_classes=5, in_channels=22, dropout_rate=0.5)

    x = torch.randn(16, 22, 750)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")