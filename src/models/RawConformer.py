"""EEG-Conformer for raw motor-imagery EEG.

Architecture (Song et al., 2023; arXiv:2106.11170):
    1. Patch-embedding stem: temporal Conv1d -> spatial Conv -> AvgPool -> dropout.
       Produces a token sequence (B, N, D).
    2. Transformer encoder: stacked pre-norm self-attention + FFN blocks.
    3. Classification head: flatten + small MLP -> num_classes.
"""

import torch
import torch.nn as nn


class _PatchEmbedding(nn.Module):
    """Temporal + spatial conv stem that emits a token sequence.

    Args:
        in_channels: Number of EEG channels.
        emb_size: Embedding (and conv channel) dim D.
        temporal_kernel: Length of the temporal Conv2d kernel along time.
        pool_kernel: AvgPool kernel along time.
        pool_stride: AvgPool stride along time.
        dropout: Dropout applied after pooling.
    """

    def __init__(self, in_channels, emb_size, temporal_kernel,
                 pool_kernel, pool_stride, dropout):
        super().__init__()
        # Treat input as (B, 1, C, T) so we can mirror the original 2D layout.
        self.temporal = nn.Conv2d(1, emb_size, kernel_size=(1, temporal_kernel),
                                  padding=(0, temporal_kernel // 2), bias=False)
        self.spatial = nn.Conv2d(emb_size, emb_size, kernel_size=(in_channels, 1),
                                 bias=False)
        self.bn = nn.BatchNorm2d(emb_size)
        self.act = nn.ELU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_kernel),
                                 stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv2d(emb_size, emb_size, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.temporal(x)         # (B, D, C, T)
        x = self.spatial(x)          # (B, D, 1, T)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)             # (B, D, 1, T')
        x = self.dropout(x)
        x = self.proj(x)             # (B, D, 1, T')
        x = x.squeeze(2).transpose(1, 2)  # (B, T', D)
        return x


class _TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, emb_size, num_heads, ff_expansion, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, num_heads,
                                          dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * ff_expansion, emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(h)
        x = x + self.ff(self.norm2(x))
        return x


class Raw_Conformer(nn.Module):
    """EEG-Conformer: conv tokenizer + transformer encoder + MLP head.

    Args:
        in_channels: EEG channels.
        num_classes: MI class count.
        n_times: Window length in samples (used to size the classifier head).
        emb_size: Token embedding dim.
        depth: Number of transformer blocks.
        num_heads: Attention heads.
        ff_expansion: FFN hidden multiplier.
        temporal_kernel: Temporal conv kernel length.
        pool_kernel: AvgPool kernel along time.
        pool_stride: AvgPool stride along time.
        dropout: Dropout used throughout.
        cls_hidden: Hidden dim of the MLP classifier.
    """

    def __init__(
            self,
            in_channels: int = 22,
            num_classes: int = 4,
            n_times: int = 750,
            emb_size: int = 40,
            depth: int = 6,
            num_heads: int = 8,
            ff_expansion: int = 4,
            temporal_kernel: int = 25,
            pool_kernel: int = 75,
            pool_stride: int = 15,
            dropout: float = 0.5,
            cls_hidden: int = 256,
    ):
        super().__init__()

        self.patch = _PatchEmbedding(
            in_channels=in_channels,
            emb_size=emb_size,
            temporal_kernel=temporal_kernel,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            dropout=dropout,
        )

        self.encoder = nn.Sequential(*[
            _TransformerBlock(emb_size, num_heads, ff_expansion, dropout)
            for _ in range(depth)
        ])

        # Compute flattened-token dim with a dry forward.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, n_times)
            tokens = self.patch(dummy)
        flat_dim = tokens.shape[1] * tokens.shape[2]

        self.classifier = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, cls_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, num_classes),
        )

    @classmethod
    def from_config(cls, cfg):
        n_times = int(cfg.preprocessing.window_sec * cfg.preprocessing.resample_rate)
        return cls(
            in_channels=cfg.eeg.in_channels,
            num_classes=cfg.eeg.num_classes,
            n_times=n_times,
            emb_size=cfg.model.emb_size,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            ff_expansion=cfg.model.ff_expansion,
            temporal_kernel=cfg.model.temporal_kernel,
            pool_kernel=cfg.model.pool_kernel,
            pool_stride=cfg.model.pool_stride,
            dropout=cfg.model.dropout,
            cls_hidden=cfg.model.cls_hidden,
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.patch(x)         # (B, N, D)
        x = self.encoder(x)       # (B, N, D)
        x = x.flatten(1)          # (B, N*D)
        return self.classifier(x)


if __name__ == "__main__":
    model = Raw_Conformer(in_channels=22, num_classes=4, n_times=750)
    out = model(torch.randn(8, 22, 750))
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
