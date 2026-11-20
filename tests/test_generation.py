import pytest
import torch

from llm_reasoning.config import ModelConfig, TokenizerConfig
from llm_reasoning.generation import generate, sample_next_token
from llm_reasoning.model import TransformerLM
from llm_reasoning.tokenizer import ByteBPETokenizer


def tiny_model(vocab_size=11):
    return TransformerLM(ModelConfig(n_layers=1, d_model=8, n_heads=2, d_ff=16,
                                     max_context_tokens=12), vocab_size=vocab_size)


def test_greedy_argmax_and_ties():
    logits = torch.tensor([[1., 5., 2.], [3., 3., -1.]])
    assert sample_next_token(logits, temperature=0, top_p=.1).tolist() == [[1], [0]]


def test_top_p_keeps_crossing_token_and_excludes_tail():
    logits = torch.tensor([[.6, .3, .1]]).log().repeat(2000, 1)
    selected = sample_next_token(logits, temperature=1, top_p=.7,
                                 generator=torch.Generator().manual_seed(42))
    assert set(selected.flatten().tolist()) == {0, 1}
    assert (sample_next_token(logits, temperature=1, top_p=.01) == 0).all()


def test_temperature_changes_distribution_and_seed_reproduces():
    logits = torch.tensor([[0., 2.]]).repeat(4000, 1)
    def sample(temp):
        return sample_next_token(logits, temperature=temp, generator=torch.Generator().manual_seed(5))
    low, high = sample(.1), sample(10.)
    assert (low == 1).float().mean() > .99
    assert .4 < (high == 1).float().mean() < .7
    assert torch.equal(high, sample(10.))


def test_real_model_greedy_matches_manual_loop_and_preserves_prompt():
    torch.manual_seed(2)
    model = tiny_model()
    ids = torch.tensor([[1, 3, 2]])
    manual = ids.clone()
    with torch.no_grad():
        for _ in range(4):
            manual = torch.cat((manual, model(manual)[:, -1].argmax(-1, keepdim=True)), dim=1)
    actual = generate(model, ids, max_new_tokens=4)
    assert torch.equal(actual, manual)
    assert ids.tolist() == [[1, 3, 2]]
    assert all(p.grad is None for p in model.parameters())
    assert model.training


def test_eos_is_included_and_stops_without_extra_forward():
    model = tiny_model()
    calls = []
    def controlled_logits(module, inputs, output):
        calls.append(inputs[0].shape[1])
        result = torch.full_like(output, -100.)
        result[..., 4 if len(calls) == 2 else 2] = 100.
        return result
    hook = model.register_forward_hook(controlled_logits)
    try:
        result = generate(model, torch.tensor([[4, 1]]), max_new_tokens=8, eos_token_id=4)
    finally:
        hook.remove()
    assert result.tolist() == [[4, 1, 2, 4]]
    assert calls == [2, 3]


def test_zero_budget_full_context_and_context_stop():
    model = tiny_model()
    prompt = torch.ones(1, 10, dtype=torch.int32)
    assert torch.equal(generate(model, prompt, max_new_tokens=0), prompt)
    assert generate(model, prompt, max_new_tokens=50).shape == (1, 12)
    full = torch.ones(1, 12, dtype=torch.long)
    assert torch.equal(generate(model, full, max_new_tokens=2), full)


def test_sampling_real_model_tokenizer_and_mixed_mode_restore():
    torch.manual_seed(9)
    tokenizer = ByteBPETokenizer(TokenizerConfig(vocab_size=300))
    model = tiny_model(tokenizer.vocab_size)
    model.blocks[0].eval()
    flags = [m.training for m in model.modules()]
    prompt = torch.tensor([tokenizer.encode('Hi')])
    def run():
        return generate(model, prompt, max_new_tokens=3, temperature=.7, top_p=.95,
                        generator=torch.Generator().manual_seed(123))
    result = run()
    assert torch.equal(result, run())
    assert result.shape == (1, 5)
    assert isinstance(tokenizer.decode(result[0].tolist()), str)
    assert [m.training for m in model.modules()] == flags


def test_modes_restored_after_forward_failure():
    model = tiny_model()
    def fail(*args):
        raise RuntimeError('test failure')
    hook = model.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match='test failure'):
            generate(model, torch.tensor([[1]]), max_new_tokens=1)
    finally:
        hook.remove()
    assert model.training and model.blocks[0].training


@pytest.mark.parametrize('kwargs', [dict(temperature=-1), dict(temperature=float('nan')),
                                  dict(top_p=0), dict(top_p=1.1)])
def test_invalid_sampling_options(kwargs):
    with pytest.raises(ValueError):
        sample_next_token(torch.zeros(1, 4), **kwargs)


def test_nonfinite_logits_fail_clearly_and_small_temperature_is_stable():
    with pytest.raises(ValueError, match='NaN'):
        sample_next_token(torch.tensor([[float('nan'), 1.]]))
    assert sample_next_token(torch.tensor([[60000., -60000.]], dtype=torch.float16),
                             temperature=1e-20).item() == 0
    assert sample_next_token(torch.tensor([[1., 0.]]), temperature=1e-300).item() == 0


def test_invalid_generation_inputs():
    model = tiny_model()
    for ids in [torch.ones(2, 2, dtype=torch.long), torch.ones(1, 0, dtype=torch.long),
                torch.ones(1, 13, dtype=torch.long), torch.ones(1, 2), torch.tensor([[-1]]),
                torch.tensor([[11]])]:
        with pytest.raises(ValueError):
            generate(model, ids, max_new_tokens=1)
    for kwargs in [dict(max_new_tokens=-1), dict(max_new_tokens=True),
                   dict(max_new_tokens=1, eos_token_id=11)]:
        with pytest.raises(ValueError):
            generate(model, torch.tensor([[1]]), **kwargs)
