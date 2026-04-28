"""A ResNet18 implementation for STFT spectrograms."""

import torch
from torch import nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention.

    Helps the model focus on relevant feature channels (frequencies/patterns)
    while suppressing noise.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Standard ResNet block with optional SE Attention."""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_se=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)  # Try Swish activation?
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_channels)

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


class STFT_ResNet18(nn.Module):
    """A Deep ResNet-18 with SE Blocks for STFT spectrograms."""

    def __init__(self, num_classes, in_channels=22, dropout_rate=0.5):
        super().__init__()

        # Initial filter bank (Stem)
        # Increase channels quickly to capture rich spatial combinations immediately
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_channels_idx = 64

        # ResNet Layers: [2, 2, 2, 2] blocks standard for ResNet-18
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

        # Weight initialization
        self._init_weights()

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        # If stride != 1 or input channels don't match output, need to adjust identity
        if stride != 1 or self.in_channels_idx != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels_idx, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(ResidualBlock(self.in_channels_idx, out_channels, stride, downsample))
        self.in_channels_idx = out_channels

        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", a=0.01, nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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
        # x: (Batch, In_Channels, H, W)
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x