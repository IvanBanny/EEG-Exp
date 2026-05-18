"""Reusable model building blocks.

Currently houses the stable-transformer components (RMSNorm, SwiGLU,
rotary attention) used by EEGEncoder, plus a causal TCN block. All
blocks operate on `(B, L, D)` sequences except `CausalTcn`, which
takes and returns the same `(B, L, D)` layout via an internal
transpose.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RmsNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019).

    Normalises by the RMS of the last dimension and applies a learned
    scale. No bias, no mean centering, lower compute than LayerNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SwigluFfn(nn.Module):
    """SwiGLU pointwise feed-forward (Shazeer, 2020).

    FFN(x) = (Swish(W_gate x) * (W_up x)) W_down.

    `hidden_dim` is the post-projection inner width. Two parallel
    projections (gate, up) produce the gating signal and the value;
    a single down-projection collapses back to `dim`.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (Su et al., 2021).

    Precomputes a `(max_seq_len, head_dim)` cos / sin lookup. Applied
    to query and key tensors of shape `(B, H, L, D_head)` via
    `apply_rotary`. `head_dim` must be even.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 1024,
                 base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2,
                                                dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (L, D/2)
        # (L, D) by interleaving pairs to match the rotation convention below
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int):
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary embeddings to (B, H, L, D_head) q and k tensors."""
    # cos / sin are (L, D_head); broadcast across (B, H)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with rotary embeddings.

    Uses `scaled_dot_product_attention` with `is_causal=True` for the
    causal mask. No KV cache (training only).
    """

    def __init__(self, dim: int, num_heads: int, max_seq_len: int,
                 attn_dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim {dim} not divisible by num_heads {num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = attn_dropout

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        b, l, d = x.shape
        q = self.q_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(l)
        q, k = apply_rotary(q, k, cos, sin)

        # PyTorch fuses the causal mask + softmax + matmul here.
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(b, l, d)
        return self.o_proj(attn_out)


class StableTransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm + causal-rotary attention,
    then RMSNorm + SwiGLU FFN, each with a residual connection."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int,
                 max_seq_len: int, attn_dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RmsNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len,
                                        attn_dropout=attn_dropout)
        self.ffn_norm = RmsNorm(dim)
        self.ffn = SwigluFfn(dim, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class StableTransformer(nn.Module):
    """Stack of `num_layers` `StableTransformerBlock`s, followed by a
    final RMSNorm. Operates on (B, L, D) sequences."""

    def __init__(self, num_layers: int, dim: int, num_heads: int,
                 ffn_dim: int, max_seq_len: int,
                 attn_dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            StableTransformerBlock(dim, num_heads, ffn_dim, max_seq_len,
                                   attn_dropout=attn_dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = RmsNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class _CausalConv1d(nn.Module):
    """1D conv with left-padding so the output stays causal.

    Pads `(kernel_size - 1) * dilation` zeros on the left, no padding
    on the right. Returns the same temporal length as the input.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class _TcnResBlock(nn.Module):
    """Two-conv causal residual block with exponential dilation.

    Each block: CausalConv1d -> BN -> ReLU -> Dropout, twice. A 1x1
    skip-connection matches channels if `in_channels != filters`.
    """

    def __init__(self, in_channels: int, filters: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()
        self.conv1 = _CausalConv1d(in_channels, filters, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(filters)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = _CausalConv1d(filters, filters, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(filters)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, filters, kernel_size=1, bias=False)
            if in_channels != filters else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = self.drop1(F.relu(self.bn1(self.conv1(x))))
        out = self.drop2(F.relu(self.bn2(self.conv2(out))))
        return F.relu(out + residual)


class CausalTcn(nn.Module):
    """Stack of `depth` causal residual blocks with exponential dilation.

    The reference EEGEncoder TCN uses kernel=4, depth=2, filters=32 and
    dilations {1, 2}. Input/output: `(B, L, D)`. Internally transposes
    to `(B, D, L)` for Conv1d. Reads the last timestep via the caller.
    """

    def __init__(self, in_dim: int, depth: int, kernel_size: int,
                 filters: int, dropout: float):
        super().__init__()
        layers = []
        prev = in_dim
        for i in range(depth):
            dilation = 2 ** i
            layers.append(_TcnResBlock(prev, filters, kernel_size, dilation, dropout))
            prev = filters
        self.blocks = nn.Sequential(*layers)
        self.out_dim = filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) -> (B, D, L)
        out = self.blocks(x.transpose(1, 2))
        return out.transpose(1, 2)
