"""Building blocks for a pre-norm decoder (task 2.1).

Residual connections and attention are assembled in later tasks. RoPE accepts
[batch, heads, sequence, head_dim] and uses adjacent pairs of coordinates.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Normalize the final dimension, with float32 statistics for half inputs."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0 or eps <= 0 or not math.isfinite(eps):
            raise ValueError("dim and finite eps must be positive")
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_floating_point() or x.shape[-1] != self.weight.numel():
            raise ValueError("RMSNorm expects floating input with matching final dimension")
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        normalized = work * torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Apply RoPE to Q/K; offset is the number of preceding cached tokens.

    Trigonometry is computed in float32 (float64 for float64 inputs). Frequencies
    are nonpersistent buffers, moved with the module and recreated from config.
    """

    def __init__(self, head_dim: int, max_context_tokens: int = 2304,
                 base: float = 10000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be positive and even")
        if max_context_tokens <= 0 or base <= 0 or not math.isfinite(base):
            raise ValueError("context and finite base must be positive")
        self.head_dim = head_dim
        self.max_context_tokens = max_context_tokens
        self.base = base
        # Integer exponents avoid rounding frequencies when .half() is called.
        self.register_buffer("frequency_indices", torch.arange(0, head_dim, 2), persistent=False)

    def forward(self, x: Tensor, *, offset: int = 0) -> Tensor:
        if x.ndim != 4 or x.shape[-1] != self.head_dim or not x.is_floating_point():
            raise ValueError("RoPE expects floating [batch, heads, sequence, head_dim]")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        length = x.shape[-2]
        if offset + length > self.max_context_tokens:
            raise ValueError("RoPE positions exceed configured context")
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        inv_freq = self.base ** (-self.frequency_indices.to(device=x.device, dtype=dtype) / self.head_dim)
        positions = torch.arange(offset, offset + length, device=x.device, dtype=dtype)
        angles = positions[:, None] * inv_freq[None, :]
        cosine, sine = angles.cos(), angles.sin()
        work = x.to(dtype)
        even, odd = work[..., 0::2], work[..., 1::2]
        return torch.stack((even * cosine - odd * sine,
                            even * sine + odd * cosine), dim=-1).flatten(-2).to(x.dtype)


class FeedForward(nn.Module):
    """Bias-free d_model -> d_ff -> d_model projection with exact GELU.

    The decoder block will call this on RMS-normalized activations (pre-norm).
    No internal normalization or residual connection is included here.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        if min(d_model, d_ff) <= 0:
            raise ValueError("feed-forward dimensions must be positive")
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.gelu(self.up_proj(x), approximate="none"))
