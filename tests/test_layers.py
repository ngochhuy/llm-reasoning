import math

import pytest
import torch

from llm_reasoning.layers import FeedForward, RMSNorm, RotaryEmbedding


def test_rmsnorm_formula_zero_and_gradients():
    layer = RMSNorm(8).double()
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    expected = x / (x.square().mean(-1, keepdim=True) + layer.eps).sqrt()
    torch.testing.assert_close(layer(x), expected)
    assert torch.count_nonzero(layer(torch.zeros_like(x))) == 0
    assert torch.autograd.gradcheck(layer, (x,))
    layer(x).square().sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(layer.weight.grad).all()


def test_rope_offset_matches_full_sequence_and_preserves_norm():
    x = torch.randn(2, 3, 11, 8, dtype=torch.float64, requires_grad=True)
    rope = RotaryEmbedding(8, max_context_tokens=16)
    full = rope(x)
    chunked = torch.cat([rope(x[:, :, :7]), rope(x[:, :, 7:], offset=7)], dim=-2)
    torch.testing.assert_close(full, chunked)
    torch.testing.assert_close(full.square().sum(-1), x.square().sum(-1))
    torch.testing.assert_close(full[:, :, 0], x[:, :, 0])
    assert torch.autograd.gradcheck(lambda t: rope(t, offset=2), (x,))


def test_rope_rotation_orientation():
    x = torch.tensor([[[[1., 0., 1., 0.]]]], dtype=torch.float64)
    actual = RotaryEmbedding(4)(x, offset=1)
    expected = torch.tensor([[[[math.cos(1), math.sin(1),
                               math.cos(.01), math.sin(.01)]]]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_norm_and_rope_preserve_dtype_with_finite_outputs(dtype):
    x = (torch.randn(2, 2, 4, 8) * 1000).to(dtype).requires_grad_()
    norm, rope = RMSNorm(8).to(dtype), RotaryEmbedding(8).to(dtype)
    y = rope(norm(x), offset=2048)
    assert y.dtype == dtype and torch.isfinite(y).all()
    y.float().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_ffn_shape_no_bias_and_pre_norm_backward():
    norm, ffn = RMSNorm(8), FeedForward(8, 32)
    x = torch.randn(2, 5, 8, requires_grad=True)
    y = x + ffn(norm(x))
    assert y.shape == x.shape
    assert all('bias' not in name for name, _ in ffn.named_parameters())
    assert sum(p.numel() for p in ffn.parameters()) == 2 * 8 * 32
    y.square().mean().backward()
    for tensor in [x, *norm.parameters(), *ffn.parameters()]:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_invalid_dimensions_and_rope_bounds():
    with pytest.raises(ValueError):
        RMSNorm(0)
    with pytest.raises(ValueError):
        FeedForward(8, 0)
    with pytest.raises(ValueError):
        RotaryEmbedding(7)
    rope = RotaryEmbedding(8, max_context_tokens=4)
    x = torch.randn(1, 1, 3, 8)
    for offset in [-1, 2, 0.5, True]:
        with pytest.raises(ValueError):
            rope(x, offset=offset)
