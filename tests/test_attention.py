import pytest
import torch
from torch.nn import functional as F

from llm_reasoning.attention import CausalSelfAttention


def test_matches_independent_pytorch_attention_reference():
    torch.manual_seed(42)
    layer = CausalSelfAttention(16, 4, max_context_tokens=12).double()
    x = torch.randn(2, 7, 16, dtype=torch.float64)
    def heads(proj):
        return proj(x).reshape(2, 7, 4, 4).transpose(1, 2)
    q, k, v = layer.rope(heads(layer.q_proj)), layer.rope(heads(layer.k_proj)), heads(layer.v_proj)
    # SDPA is a test oracle only; production code computes attention explicitly.
    reference = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    reference = layer.out_proj(reference.transpose(1, 2).reshape(2, 7, 16))
    torch.testing.assert_close(layer(x), reference)
    assert sum(p.numel() for p in layer.parameters()) == 4 * 16 * 16
    assert all('bias' not in name for name, _ in layer.named_parameters())


def test_future_tokens_do_not_change_prefix_outputs_or_gradients():
    torch.manual_seed(7)
    layer = CausalSelfAttention(16, 2).double()
    x = torch.randn(2, 8, 16, dtype=torch.float64, requires_grad=True)
    changed = x.detach().clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 30
    full = layer(x)
    torch.testing.assert_close(full[:, :4], layer(changed)[:, :4])
    torch.testing.assert_close(full[:, :4], layer(x[:, :4]))
    full[:, :4].square().sum().backward()
    assert torch.count_nonzero(x.grad[:, 4:]) == 0
    assert x.grad[:, :4].abs().sum() > 0


def test_shape_single_token_and_backward():
    layer = CausalSelfAttention(8, 2, max_context_tokens=8)
    x = torch.randn(2, 1, 8, requires_grad=True)
    # At length one, attention probability must be exactly one.
    y = layer(x)
    torch.testing.assert_close(y, layer.out_proj(layer.v_proj(x)))
    assert y.shape == x.shape
    y.square().sum().backward()
    for parameter in layer.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert torch.isfinite(x.grad).all()


def test_attention_numerical_gradient():
    layer = CausalSelfAttention(4, 1).double()
    x = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (x,))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_large_logits_finite_in_low_precision(dtype):
    torch.manual_seed(9)
    layer = CausalSelfAttention(8, 2).to(dtype)
    x = (torch.randn(1, 6, 8) * 1000).to(dtype).requires_grad_()
    y = layer(x)
    assert y.dtype == dtype and torch.isfinite(y).all()
    y.float().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


def test_cpu_autocast_keeps_scores_stable():
    layer = CausalSelfAttention(8, 2)
    x = torch.randn(1, 5, 8, requires_grad=True) * 1000
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y = layer(x)
    assert y.dtype == torch.bfloat16 and torch.isfinite(y).all()
    y.float().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters())


def test_rejects_invalid_dimensions_and_sequences():
    for dims in [(0, 2), (8, 0), (8, 3), (6, 2)]:
        with pytest.raises(ValueError):
            CausalSelfAttention(*dims)
    layer = CausalSelfAttention(8, 2, max_context_tokens=4)
    for x in [torch.randn(1, 0, 8), torch.randn(1, 5, 8), torch.randn(2, 8),
              torch.randn(1, 2, 4), torch.ones(1, 2, 8, dtype=torch.long)]:
        with pytest.raises(ValueError):
            layer(x)
