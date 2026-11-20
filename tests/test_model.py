from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from llm_reasoning.config import ModelConfig
from llm_reasoning.model import DecoderBlock, TransformerLM


def small_config():
    return ModelConfig(n_layers=2, d_model=16, n_heads=4, d_ff=32, max_context_tokens=16)


def test_decoder_block_uses_pre_norm_residuals():
    block = DecoderBlock(small_config())
    x = torch.randn(2, 5, 16)
    first = x + block.attention(block.attention_norm(x))
    expected = first + block.ffn(block.ffn_norm(first))
    torch.testing.assert_close(block(x), expected)


@pytest.mark.parametrize('tied', [True, False])
def test_model_shapes_shifted_loss_and_all_parameter_gradients(tied):
    torch.manual_seed(42)
    model = TransformerLM(replace(small_config(), tie_embeddings=tied), vocab_size=37)
    tokens = torch.randint(0, 37, (2, 8))
    logits = model(tokens)
    assert logits.shape == (2, 8, 37) and torch.isfinite(logits).all()
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 37), tokens[:, 1:].reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert (model.lm_head.weight is model.token_embedding.weight) is tied
    cfg = model.config
    expected = 37*16 + cfg.n_layers*(4*16*16 + 2*16*32 + 2*16) + 16
    if not tied:
        expected += 37*16
    assert model.num_parameters() == expected
    assert all('bias' not in name for name, _ in model.named_parameters())


def test_model_preserves_causality_across_all_blocks():
    torch.manual_seed(10)
    model = TransformerLM(small_config(), vocab_size=37).eval()
    tokens = torch.randint(0, 37, (2, 9))
    changed = tokens.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 37
    with torch.no_grad():
        original = model(tokens)
        torch.testing.assert_close(original[:, :4], model(changed)[:, :4])
        torch.testing.assert_close(original[:, :4], model(tokens[:, :4]))


def test_state_dict_roundtrip_and_seeded_initialization():
    torch.manual_seed(20)
    model = TransformerLM(small_config(), vocab_size=37)
    torch.manual_seed(20)
    same_seed = TransformerLM(small_config(), vocab_size=37)
    for a, b in zip(model.parameters(), same_seed.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    restored = TransformerLM(small_config(), vocab_size=37)
    restored.load_state_dict(model.state_dict())
    tokens = torch.randint(0, 37, (1, 5))
    torch.testing.assert_close(restored(tokens), model(tokens), rtol=0, atol=0)
    assert restored.lm_head.weight is restored.token_embedding.weight


def test_target_model_parameter_count_and_short_forward_backward():
    # Use the actual target geometry and vocabulary; only sequence length is small.
    model = TransformerLM(ModelConfig(), vocab_size=8000)
    assert len(model.blocks) == 10
    assert model.num_parameters() == 35_564_032
    ids = torch.randint(0, 8000, (1, 8))
    logits = model(ids)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 8000), ids[:, 1:].reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_model_bfloat16_cpu_autocast():
    model = TransformerLM(small_config(), vocab_size=37)
    ids = torch.randint(0, 37, (1, 6))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        logits = model(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 37), ids[:, 1:].reshape(-1))
    assert logits.dtype == torch.bfloat16 and torch.isfinite(logits).all()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_invalid_input_and_context_boundary():
    for vocab_size in [0, -1, True, 3.5]:
        with pytest.raises(ValueError):
            TransformerLM(small_config(), vocab_size=vocab_size)
    model = TransformerLM(small_config(), vocab_size=37)
    assert model(torch.zeros(1, 16, dtype=torch.long)).shape == (1, 16, 37)
    for ids in [torch.zeros(1, 17, dtype=torch.long), torch.zeros(0, 3, dtype=torch.long),
                torch.zeros(1, 0, dtype=torch.long), torch.zeros(1, 3), torch.zeros(3, dtype=torch.long)]:
        with pytest.raises(ValueError):
            model(ids)
    with pytest.raises((IndexError, RuntimeError)):
        model(torch.tensor([[37]]))
