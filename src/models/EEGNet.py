"""EEGNet (Lawhern et al., 2018) for raw motor-imagery EEG.

Compact CNN with three stages:
    1. Temporal Conv2d -> BN
    2. Depthwise Conv2d across electrodes (per-temporal-filter spatial filter)
       with max-norm constraint -> BN -> ELU -> AvgPool -> Dropout
    3. Separable Conv2d (depthwise-temporal + pointwise) -> BN -> ELU -> AvgPool
       -> Dropout
    4. Flatten -> Dense classifier

Treats input as (B, 1, C, T). Total parameter count is small (typically a few
thousand) which makes EEGNet a strong, well-regularized baseline on small MI
datasets.
"""

import torch
import torch.nn as nn


def _max_norm(module, max_val):
    """In-place max-norm clamp on the module's weight along output dim."""
    with torch.no_grad():
        w = module.weight
        flat = w.view(w.shape[0], -1)
        norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        desired = norm.clamp(max=max_val)
        flat.mul_(desired / norm)


class _MaxNormHook:
    """Forward-pre-hook that re-applies max-norm to a module's weight."""

    def __init__(self, max_val):
        self.max_val = max_val

    def __call__(self, module, _inputs):
        _max_norm(module, self.max_val)


class EEGNet(nn.Module):
    """EEGNet v2 (8,2) style architecture, parameterized for arbitrary inputs.

    Args:
        in_channels: Number of EEG channels.
        num_classes: Number of classes.
        n_times: Window length in samples (used for the FC head).
        f1: Number of temporal filters in the first stage.
        d: Depth multiplier for the depthwise spatial filter.
        f2: Number of pointwise filters (typically f1 * d).
        kernel_length: Temporal conv kernel length.
        pool1: AvgPool kernel along time after the depthwise stage.
        pool2: AvgPool kernel along time after the separable stage.
        sep_kernel: Separable depthwise temporal kernel.
        dropout: Dropout rate.
        depth_max_norm: Max-norm bound on the depthwise spatial conv weights.
        cls_max_norm: Max-norm bound on the classifier weights.
    """

    def __init__(
            self,
            in_channels: int = 22,
            num_classes: int = 4,
            n_times: int = 750,
            f1: int = 8,
            d: int = 2,
            f2: int = 16,
            kernel_length: int = 64,
            pool1: int = 4,
            pool2: int = 8,
            sep_kernel: int = 16,
            dropout: float = 0.5,
            depth_max_norm: float = 1.0,
            cls_max_norm: float = 0.25,
            endpool_ms: float = None,
            sfreq: float = 250.0,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, kernel_length),
                      padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
        )

        self.depthwise = nn.Conv2d(
            f1, f1 * d, kernel_size=(in_channels, 1),
            groups=f1, bias=False,
        )
        self.depthwise.register_forward_pre_hook(_MaxNormHook(depth_max_norm))

        self.block2 = nn.Sequential(
            nn.BatchNorm2d(f1 * d),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, pool1)),
            nn.Dropout(dropout),
        )

        self.separable = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, kernel_size=(1, sep_kernel),
                      padding=(0, sep_kernel // 2), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, pool2)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_channels, n_times)
            feat = self.separable(self.block2(self.depthwise(self.block1(dummy))))
        # feat shape: (1, f2, 1, t_feat)
        t_feat = feat.shape[-1]

        # Deployment-honest end-of-window pooling: average over the most-recent
        # `endpool_ms` of the input window. We translate ms-of-input into
        # frames-of-feature-map via the cumulative time stride of the two pools
        # (block2 / separable each downsample by pool1 / pool2 respectively).
        # endpool_ms=None -> legacy full-flatten behaviour.
        self.endpool_ms = endpool_ms
        if endpool_ms is None:
            self.endpool_frames = None
            flat_dim = feat.numel()
        else:
            time_stride = pool1 * pool2
            k_input = max(1, int(round(endpool_ms * 1e-3 * sfreq)))
            k_feat = max(1, min(t_feat, int(round(k_input / time_stride))))
            self.endpool_frames = k_feat
            flat_dim = f2

        self.classifier = nn.Linear(flat_dim, num_classes)
        self.classifier.register_forward_pre_hook(_MaxNormHook(cls_max_norm))

    @classmethod
    def from_config(cls, cfg):
        n_times = int(cfg.preprocessing.window_sec * cfg.preprocessing.resample_rate)
        return cls(
            in_channels=cfg.eeg.in_channels,
            num_classes=cfg.eeg.num_classes,
            n_times=n_times,
            f1=cfg.model.f1,
            d=cfg.model.d,
            f2=cfg.model.f2,
            kernel_length=cfg.model.kernel_length,
            pool1=cfg.model.pool1,
            pool2=cfg.model.pool2,
            sep_kernel=cfg.model.sep_kernel,
            dropout=cfg.model.dropout,
            endpool_ms=cfg.model.get("endpool_ms", None),
            sfreq=float(cfg.preprocessing.resample_rate),
        )

    def forward(self, x):
        # x: (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.block2(x)
        x = self.separable(x)
        # x: (B, f2, 1, t_feat)
        if self.endpool_frames is None:
            x = x.flatten(1)
        else:
            x = x[..., -self.endpool_frames:].mean(dim=-1)  # (B, f2, 1)
            x = x.flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    m = EEGNet(in_channels=22, num_classes=4, n_times=750)
    out = m(torch.randn(8, 22, 750))
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in m.parameters()):,}")
