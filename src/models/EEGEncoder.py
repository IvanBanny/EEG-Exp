"""EEGEncoder (Liao & Wang, 2024) for raw motor-imagery EEG.

1. Downsampling projector. Three Conv2d stages with batch norm, ELU,
   average pooling and dropout. Collapses (B, 1, C, T) into a short
   sequence (B, L', F2) where L' is roughly T / (pool1 * pool2).
2. n_branches parallel "Dual-Stream Temporal-Spatial" (DSTS) blocks.
   Each branch dropouts the projector output, then runs both a causal
   TCN and a small "stable" transformer (RMSNorm + SwiGLU + rotary
   embeddings + causal mask). The TCN reads its last timestep, the
   transformer takes its mean over time, the two are summed (with
   dropout on the transformer branch), and a linear head produces
   per-class logits. The branch outputs are averaged.

The model returns raw logits; the trainer applies CE + label smoothing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._blocks import CausalTcn, StableTransformer


class DownsamplingProjector(nn.Module):
    """Three-stage Conv2d downsampler.

    Mirrors `ConvBlock` in the reference. The same-padding stages keep
    the temporal axis intact across the convs, and two AvgPool stages
    cut sequence length by `pool1 * pool2`.

    Args:
        in_channels: Number of EEG channels (collapsed in stage 2).
        f1: Temporal filter count for stage 1.
        d: Depth multiplier for the depthwise spatial conv.
        kernel_length: Temporal conv kernel for stage 1.
        sep_kernel: Pointwise temporal conv kernel for stage 3.
        pool1: AvgPool kernel after the depthwise stage.
        pool2: AvgPool kernel after the separable stage.
        dropout: Dropout rate applied after each pool.
    """

    def __init__(self, in_channels: int = 22, f1: int = 16, d: int = 2,
                 kernel_length: int = 64, sep_kernel: int = 16,
                 pool1: int = 8, pool2: int = 7, dropout: float = 0.3):
        super().__init__()
        f2 = f1 * d
        self.out_dim = f2

        # Stage 1: temporal conv (no activation, paper convention).
        # `padding='same'` keeps T intact across even-kernel temporal convs.
        self.conv1 = nn.Conv2d(
            1, f1, kernel_size=(kernel_length, 1),
            padding="same", bias=False,
        )
        self.bn1 = nn.BatchNorm2d(f1)

        # Stage 2: depthwise spatial collapse + pool
        self.depthwise = nn.Conv2d(
            f1, f2, kernel_size=(1, in_channels), groups=f1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(f2)
        self.pool1 = nn.AvgPool2d(kernel_size=(pool1, 1))
        self.drop1 = nn.Dropout(dropout)

        # Stage 3: pointwise temporal conv + pool
        self.conv2 = nn.Conv2d(
            f2, f2, kernel_size=(sep_kernel, 1),
            padding="same", bias=False,
        )
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d(kernel_size=(pool2, 1))
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, 1, T, C)
        x = x.unsqueeze(1).transpose(2, 3)

        x = self.bn1(self.conv1(x))
        x = F.elu(self.bn2(self.depthwise(x)))
        x = self.drop1(self.pool1(x))
        x = F.elu(self.bn3(self.conv2(x)))
        x = self.drop2(self.pool2(x))

        # x: (B, F2, L', 1) -> (B, L', F2)
        return x.squeeze(-1).transpose(1, 2)


class DstsBranch(nn.Module):
    """One parallel branch: TCN + stable transformer + linear head."""

    def __init__(self, dim: int, num_classes: int, num_heads: int,
                 num_layers: int, ffn_dim: int, max_seq_len: int,
                 tcn_depth: int, tcn_kernel: int, tcn_filters: int,
                 tcn_dropout: float, branch_dropout: float,
                 trm_dropout: float):
        super().__init__()
        self.branch_drop = nn.Dropout(branch_dropout)
        self.tcn = CausalTcn(
            in_dim=dim, depth=tcn_depth, kernel_size=tcn_kernel,
            filters=tcn_filters, dropout=tcn_dropout,
        )
        self.trm = StableTransformer(
            num_layers=num_layers, dim=dim, num_heads=num_heads,
            ffn_dim=ffn_dim, max_seq_len=max_seq_len,
        )
        if self.tcn.out_dim != dim:
            raise ValueError(
                f"tcn.out_dim={self.tcn.out_dim} must equal dim={dim} so "
                f"the TCN last-timestep and the transformer mean can be summed"
            )
        self.trm_dropout = trm_dropout
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        x = self.branch_drop(x)
        tcn_last = self.tcn(x)[:, -1, :]  # (B, F)
        trm_mean = self.trm(x).mean(dim=1)  # (B, D)
        fused = tcn_last + F.dropout(
            trm_mean, p=self.trm_dropout, training=self.training,
        )
        return self.head(fused)


class EEGEncoder(nn.Module):
    """Stacked downsampler + n_branches parallel DSTS branches.

    The branch outputs are averaged. Returns raw logits.

    Args:
        in_channels: Number of EEG channels.
        num_classes: Number of MI classes.
        n_times: Number of input time samples per window.
        f1: Stage 1 temporal filter count (paper default 16).
        d: Depthwise spatial depth multiplier (paper default 2).
        kernel_length: Stage 1 temporal kernel (paper default 64).
        sep_kernel: Stage 3 pointwise temporal kernel (paper default 16).
        pool1: Pool after stage 2 (paper default 8).
        pool2: Pool after stage 3 (paper default 7).
        projector_dropout: Dropout in the projector (paper default 0.3).
        n_branches: Number of parallel DSTS branches (paper default 5).
        num_heads: Transformer attention heads (paper code default 2).
        num_layers: Transformer layers (paper code default 2).
        ffn_dim: Transformer SwiGLU inner dimension (paper code default
            = F2 = f1 * d).
        tcn_depth: Causal TCN depth (paper default 2).
        tcn_kernel: TCN kernel size (paper default 4).
        tcn_filters: TCN filter count (paper default 32 = F2).
        tcn_dropout: Dropout inside the TCN (paper default 0.3).
        branch_dropout: Dropout on the projector output per branch.
        trm_dropout: Dropout on the transformer output before fusion.
    """

    def __init__(
            self,
            in_channels: int = 22,
            num_classes: int = 4,
            n_times: int = 1125,
            f1: int = 16,
            d: int = 2,
            kernel_length: int = 64,
            sep_kernel: int = 16,
            pool1: int = 8,
            pool2: int = 7,
            projector_dropout: float = 0.3,
            n_branches: int = 5,
            num_heads: int = 2,
            num_layers: int = 2,
            ffn_dim: int = None,
            tcn_depth: int = 2,
            tcn_kernel: int = 4,
            tcn_filters: int = 32,
            tcn_dropout: float = 0.3,
            branch_dropout: float = 0.3,
            trm_dropout: float = 0.3,
    ):
        super().__init__()
        f2 = f1 * d
        if ffn_dim is None:
            ffn_dim = f2
        if tcn_filters != f2:
            raise ValueError(
                f"tcn_filters ({tcn_filters}) must equal F2=f1*d ({f2}) so "
                f"TCN output dimension matches the transformer dimension"
            )

        self.projector = DownsamplingProjector(
            in_channels=in_channels, f1=f1, d=d,
            kernel_length=kernel_length, sep_kernel=sep_kernel,
            pool1=pool1, pool2=pool2, dropout=projector_dropout,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, n_times)
            seq = self.projector(dummy)
        max_seq_len = max(seq.shape[1], 1)
        if seq.shape[1] < 2:
            raise ValueError(
                f"Projector output sequence length is {seq.shape[1]} for "
                f"n_times={n_times} with pool1={pool1}, pool2={pool2}. "
                f"Increase n_times or reduce the pool factors."
            )

        self.branches = nn.ModuleList([
            DstsBranch(
                dim=f2, num_classes=num_classes,
                num_heads=num_heads, num_layers=num_layers,
                ffn_dim=ffn_dim, max_seq_len=max_seq_len,
                tcn_depth=tcn_depth, tcn_kernel=tcn_kernel,
                tcn_filters=tcn_filters, tcn_dropout=tcn_dropout,
                branch_dropout=branch_dropout, trm_dropout=trm_dropout,
            )
            for _ in range(n_branches)
        ])

    @classmethod
    def from_config(cls, cfg):
        n_times = int(cfg.preprocessing.window_sec * cfg.preprocessing.resample_rate)
        return cls(
            in_channels=cfg.eeg.in_channels,
            num_classes=cfg.eeg.num_classes,
            n_times=n_times,
            f1=cfg.model.f1,
            d=cfg.model.d,
            kernel_length=cfg.model.kernel_length,
            sep_kernel=cfg.model.sep_kernel,
            pool1=cfg.model.pool1,
            pool2=cfg.model.pool2,
            projector_dropout=cfg.model.projector_dropout,
            n_branches=cfg.model.n_branches,
            num_heads=cfg.model.num_heads,
            num_layers=cfg.model.num_layers,
            ffn_dim=cfg.model.get("ffn_dim", None),
            tcn_depth=cfg.model.tcn_depth,
            tcn_kernel=cfg.model.tcn_kernel,
            tcn_filters=cfg.model.tcn_filters,
            tcn_dropout=cfg.model.tcn_dropout,
            branch_dropout=cfg.model.branch_dropout,
            trm_dropout=cfg.model.trm_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        seq = self.projector(x)
        logits = torch.stack([branch(seq) for branch in self.branches], dim=0)
        return logits.mean(dim=0)


if __name__ == "__main__":
    m = EEGEncoder(in_channels=22, num_classes=4, n_times=1125)
    out = m(torch.randn(8, 22, 1125))
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in m.parameters()):,}")
