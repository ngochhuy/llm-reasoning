"""Explicit multi-head causal self-attention, without a packaged attention layer."""

import torch
from torch import Tensor, nn

from llm_reasoning.layers import RotaryEmbedding
from llm_reasoning.kv_cache import LayerKVCache


class CausalSelfAttention(nn.Module):
    """Bias-free MHA for [batch, sequence, d_model], with RoPE on Q and K.

    Inputs represent contiguous sequences, optionally right-padded by the future
    dataloader. Padding loss must be masked by the training loop. No cross-document
    isolation, left-padding support or dropout is implemented here.
    """

    def __init__(self, d_model: int, n_heads: int,
                 max_context_tokens: int = 2304, rope_base: float = 10000.0) -> None:
        super().__init__()
        if min(d_model, n_heads) <= 0 or d_model % n_heads:
            raise ValueError("d_model must be positive and divisible by positive n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = RotaryEmbedding(self.head_dim, max_context_tokens, rope_base)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, *, cache: LayerKVCache | None = None) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d_model or not x.is_floating_point():
            raise ValueError("attention expects floating [batch, sequence, d_model]")
        batch, length, _ = x.shape
        if length == 0 or length > self.rope.max_context_tokens:
            raise ValueError("sequence length must be positive and within context")

        def split_heads(projection: nn.Linear) -> Tensor:
            return projection(x).reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        offset = 0 if cache is None else cache.length
        q = self.rope(split_heads(self.q_proj), offset=offset)
        k = self.rope(split_heads(self.k_proj), offset=offset)
        v = split_heads(self.v_proj)
        if cache is not None:
            k, v = cache.append(k, v)
        # Disable ambient autocast here: casting tensors alone does not prevent
        # matmul from being autocast back to half and overflowing its logits.
        work_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = (q.to(work_dtype) @ k.to(work_dtype).transpose(-2, -1)) * self.head_dim**-0.5
            query_positions = torch.arange(offset, offset + length, device=x.device)
            key_positions = torch.arange(k.shape[-2], device=x.device)
            future = key_positions[None, :] > query_positions[:, None]
            scores = scores.masked_fill(future, float("-inf"))
            probabilities = torch.softmax(scores, dim=-1)
            context = probabilities @ v.to(work_dtype)
        merged = context.to(v.dtype).transpose(1, 2).reshape(batch, length, self.d_model)
        return self.out_proj(merged)
