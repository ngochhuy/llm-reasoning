"""Decoder-only Transformer assembled from the project's own PyTorch blocks."""

import math

import torch
from torch import Tensor, nn

from llm_reasoning.attention import CausalSelfAttention
from llm_reasoning.config import ModelConfig
from llm_reasoning.layers import FeedForward, RMSNorm
from llm_reasoning.kv_cache import KVCache, LayerKVCache


class DecoderBlock(nn.Module):
    """Two pre-norm residual branches: causal attention, then GELU FFN."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(
            config.d_model, config.n_heads, config.max_context_tokens,
        )
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = FeedForward(config.d_model, config.d_ff)

    def forward(self, x: Tensor, *, cache: LayerKVCache | None = None) -> Tensor:
        x = x + self.attention(self.attention_norm(x), cache=cache)
        return x + self.ffn(self.ffn_norm(x))


class TransformerLM(nn.Module):
    """Return unnormalized logits [batch, sequence, vocab_size].

    Vocabulary size must come from the saved tokenizer, not an assumed target.
    With a cache, input_ids contains only the new chunk (full prompt for prefill).
    RoPE lives inside attention; there are no learned positional embeddings.
    """

    def __init__(self, config: ModelConfig, vocab_size: int) -> None:
        super().__init__()
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")
        self.config = config
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        # Initialize the shared embedding/head only once. Scale residual output
        # projections to control variance across the 2*n_layers residual branches.
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.normal_(block.attention.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=residual_std)
        if not self.config.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, *, cache: KVCache | None = None) -> Tensor:
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be int32/int64 [batch, sequence]")
        if input_ids.shape[0] == 0 or not 0 < input_ids.shape[1] <= self.config.max_context_tokens:
            raise ValueError("batch must be nonempty and sequence length within context")
        staged = None
        if cache is not None:
            if torch.is_grad_enabled() or self.training:
                raise ValueError("cache requires model.eval() and no_grad/inference_mode")
            if len(cache.layers) != len(self.blocks) or cache.model_id not in (None, id(self)):
                raise ValueError("cache belongs to another model or layer configuration")
            if cache.length + input_ids.shape[1] > self.config.max_context_tokens:
                raise ValueError("cached prefix plus new tokens exceeds context")
            staged = cache.fork()
        hidden = self.token_embedding(input_ids)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, cache=None if staged is None else staged.layers[index])
        logits = self.lm_head(self.final_norm(hidden))
        if staged is not None:
            cache.layers = staged.layers
            cache.model_id = id(self)
        return logits

    def num_parameters(self) -> int:
        """Count unique parameters; the tied LM head is not counted twice."""
        return sum(parameter.numel() for parameter in self.parameters())
