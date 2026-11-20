"""Dynamic tensor KV storage, owned by a single inference request."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LayerKVCache:
    key: Tensor | None = None
    value: Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.key is None else self.key.shape[-2]

    def append(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        if torch.is_grad_enabled():
            raise ValueError("KV cache is inference-only; use no_grad or inference_mode")
        if key.ndim != 4 or key.shape != value.shape or key.shape[-2] == 0:
            raise ValueError("K/V must have equal [batch, heads, nonempty sequence, head_dim] shapes")
        if key.dtype != value.dtype or key.device != value.device:
            raise ValueError("K/V dtype and device must match")
        if self.key is not None:
            old = self.key
            if (old.shape[:2] != key.shape[:2] or old.shape[-1] != key.shape[-1]
                    or old.dtype != key.dtype or old.device != key.device):
                raise ValueError("cache batch/head geometry, dtype or device changed; reset cache")
            key = torch.cat((old, key), dim=-2)
            value = torch.cat((self.value, value), dim=-2)
        self.key, self.value = key, value
        return key, value


class KVCache:
    """Store RoPE-rotated K and raw V for each decoder layer.

    Create a fresh cache or reset between prompts. Reset also releases tensor
    references; it does not empty the CUDA allocator's reserved memory.
    """

    def __init__(self, n_layers: int) -> None:
        if not isinstance(n_layers, int) or n_layers <= 0:
            raise ValueError("n_layers must be positive")
        self.layers = [LayerKVCache() for _ in range(n_layers)]
        self.model_id: int | None = None

    @property
    def length(self) -> int:
        lengths = {layer.length for layer in self.layers}
        if len(lengths) != 1:
            raise ValueError("cache layers have inconsistent lengths; reset cache")
        return next(iter(lengths))

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for layer in self.layers
                   for t in (layer.key, layer.value) if t is not None)

    def reset(self) -> None:
        self.layers = [LayerKVCache() for _ in self.layers]
        self.model_id = None

    def fork(self) -> "KVCache":
        """Stage updates without changing this cache if a forward call fails."""
        result = KVCache(len(self.layers))
        result.layers = [LayerKVCache(layer.key, layer.value) for layer in self.layers]
        result.model_id = self.model_id
        return result
