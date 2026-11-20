"""Autoregressive decoding with optional per-request KV cache."""

import math

import torch
from torch import Tensor

from llm_reasoning.model import TransformerLM
from llm_reasoning.kv_cache import KVCache


def _validate_sampling(temperature: float, top_p: float) -> None:
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative (0 = greedy)")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")


def sample_next_token(logits: Tensor, *, temperature: float = 0.0,
                      top_p: float = 1.0, generator: torch.Generator | None = None) -> Tensor:
    """Select [batch, 1] token IDs from finite [batch, vocabulary] logits.

    Temperature zero is greedy; otherwise sample the smallest sorted nucleus
    whose cumulative probability reaches top_p, including the crossing token.
    The optional generator must be on the same device as logits.
    """
    _validate_sampling(temperature, top_p)
    if logits.ndim != 2 or min(logits.shape) == 0 or not logits.is_floating_point():
        raise ValueError("logits must be floating [nonempty batch, nonempty vocabulary]")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain NaN or infinity")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    work = logits.double() if logits.dtype == torch.float64 or temperature < torch.finfo(torch.float32).tiny else logits.float()
    # Subtract max before scaling to avoid positive overflow at small temperatures.
    probabilities = torch.softmax((work - work.max(dim=-1, keepdim=True).values) / temperature, dim=-1)
    if top_p < 1:
        sorted_probs, indices = probabilities.sort(dim=-1, descending=True)
        remove = sorted_probs.cumsum(dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0)
        selected = torch.multinomial(sorted_probs, 1, generator=generator)
        return indices.gather(-1, selected)
    return torch.multinomial(probabilities, 1, generator=generator)


@torch.inference_mode()
def generate(model: TransformerLM, input_ids: Tensor, *, max_new_tokens: int,
             eos_token_id: int | None = None, temperature: float = 0.0,
             top_p: float = 1.0, generator: torch.Generator | None = None,
             use_cache: bool = False) -> Tensor:
    """Generate one unpadded prompt [1, T]; return prompt plus new tokens.

    Stop at the first newly generated EOS (included), max_new_tokens, or context
    capacity. Never crop the prompt or reset RoPE positions. With use_cache=True,
    prefill once then forward just one token; a fresh cache is used for each call.
    """
    _validate_sampling(temperature, top_p)
    if not isinstance(use_cache, bool):
        raise ValueError("use_cache must be boolean")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must be int32/int64 [1, sequence], with no padding")
    length = input_ids.shape[1]
    if not 0 < length <= model.config.max_context_tokens:
        raise ValueError("prompt must be nonempty and within context capacity")
    if input_ids.device != model.token_embedding.weight.device:
        raise ValueError("prompt and model must be on the same device")
    if (input_ids < 0).any() or (input_ids >= model.vocab_size).any():
        raise ValueError("prompt token ID outside model vocabulary")
    if eos_token_id is not None and (
        not isinstance(eos_token_id, int) or isinstance(eos_token_id, bool)
        or not 0 <= eos_token_id < model.vocab_size
    ):
        raise ValueError("eos_token_id must be a valid vocabulary ID or None")
    tokens = input_ids.clone()
    steps = min(max_new_tokens, model.config.max_context_tokens - length)
    # Preserve mixed train/eval flags too, including if a forward call fails.
    training_flags = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        cache = KVCache(len(model.blocks)) if use_cache else None
        for step in range(steps):
            chunk = tokens if cache is None or step == 0 else tokens[:, -1:]
            logits = (model(chunk) if cache is None else model(chunk, cache=cache))[:, -1, :]
            next_token = sample_next_token(logits, temperature=temperature, top_p=top_p, generator=generator)
            tokens = torch.cat((tokens, next_token.to(tokens.dtype)), dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    finally:
        for module, was_training in training_flags:
            module.training = was_training
    return tokens
