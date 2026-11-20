import pytest
import torch

from llm_reasoning.config import ModelConfig
from llm_reasoning.generation import generate
from llm_reasoning.kv_cache import KVCache, LayerKVCache
from llm_reasoning.model import TransformerLM


def model_for_test(context=32, layers=2):
    return TransformerLM(ModelConfig(n_layers=layers, d_model=16, n_heads=4,
                                     d_ff=32, max_context_tokens=context), 43).eval()


@pytest.mark.parametrize('seed', [0, 7, 42])
@pytest.mark.parametrize('chunks', [[6, 1, 1, 1], [2, 3, 4], [1]*9])
def test_chunked_cache_logits_equal_full_forward(seed, chunks):
    torch.manual_seed(seed)
    model = model_for_test()
    ids = torch.randint(0, 43, (2, sum(chunks)))
    cache = KVCache(len(model.blocks))
    with torch.inference_mode():
        expected = model(ids)
        outputs, start = [], 0
        for length in chunks:
            outputs.append(model(ids[:, start:start+length], cache=cache))
            start += length
            assert cache.length == start
        actual = torch.cat(outputs, dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for layer in cache.layers:
        assert layer.key.shape == layer.value.shape == (2, 4, 9, 4)
        assert not layer.key.requires_grad and not layer.value.requires_grad
    assert cache.nbytes == 2 * 2 * 2 * 4 * 9 * 4 * 4


@pytest.mark.parametrize('seed', [0, 7, 42])
def test_greedy_cache_matches_and_requests_are_independent(seed):
    torch.manual_seed(seed)
    model = model_for_test()
    model.train()  # generate must enter eval and restore train mode.
    prompt = torch.randint(0, 43, (1, 7))
    off = generate(model, prompt, max_new_tokens=8, use_cache=False)
    on = generate(model, prompt, max_new_tokens=8, use_cache=True)
    assert torch.equal(on, off)
    assert torch.equal(on, generate(model, prompt, max_new_tokens=8, use_cache=True))
    assert model.training


def test_cached_generation_prefills_once_then_forwards_single_tokens():
    model = model_for_test()
    sizes = []
    hook = model.register_forward_pre_hook(lambda module, args: sizes.append(args[0].shape[1]))
    try:
        generate(model, torch.tensor([[1, 2, 3, 4]]), max_new_tokens=5, use_cache=True)
        assert sizes == [4, 1, 1, 1, 1]
        sizes.clear()
        generate(model, torch.tensor([[1, 2, 3, 4]]), max_new_tokens=5, use_cache=False)
        assert sizes == [4, 5, 6, 7, 8]
    finally:
        hook.remove()


def test_reset_releases_tensors_and_restarts_rope_positions():
    model = model_for_test()
    cache = KVCache(2)
    with torch.no_grad():
        model(torch.tensor([[1, 2, 3]]), cache=cache)
        cache.reset()
        assert cache.length == cache.nbytes == 0 and cache.model_id is None
        assert all(layer.key is None and layer.value is None for layer in cache.layers)
        ids = torch.tensor([[8, 5]])
        torch.testing.assert_close(model(ids, cache=cache), model(ids))
        assert cache.length == 2


def test_context_limit_and_failure_do_not_partially_update_cache():
    model = model_for_test(context=8)
    cache = KVCache(2)
    with torch.no_grad():
        model(torch.tensor([[1, 2, 3, 4, 5, 6, 7]]), cache=cache)
        model(torch.tensor([[8]]), cache=cache)
        assert cache.length == 8
        with pytest.raises(ValueError, match='context'):
            model(torch.tensor([[9]]), cache=cache)
        assert cache.length == 8
        cache.reset()
        model(torch.tensor([[1, 2]]), cache=cache)
        original = [layer.key for layer in cache.layers]
        def fail(*args):
            raise RuntimeError('injected failure')
        hook = model.blocks[1].register_forward_pre_hook(fail)
        try:
            with pytest.raises(RuntimeError, match='injected'):
                model(torch.tensor([[3]]), cache=cache)
        finally:
            hook.remove()
        assert cache.length == 2
        assert all(layer.key is old for layer, old in zip(cache.layers, original))
        torch.testing.assert_close(model(torch.tensor([[3]]), cache=cache),
                                   model(torch.tensor([[1, 2, 3]]))[:, -1:], rtol=1e-5, atol=1e-6)


def test_cache_rejects_training_wrong_model_and_batch():
    model = model_for_test()
    cache = KVCache(2)
    ids = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match='no_grad'):
        model(ids, cache=cache)
    with torch.no_grad():
        model.train()
        with pytest.raises(ValueError, match='eval'):
            model(ids, cache=cache)
        model.eval()
        with pytest.raises(ValueError, match='layer'):
            model(ids, cache=KVCache(1))
        model(ids, cache=cache)
        with pytest.raises(ValueError, match='another model'):
            model_for_test()(ids, cache=cache)
        with pytest.raises(ValueError, match='geometry'):
            model(torch.ones(2, 1, dtype=torch.long), cache=cache)
        assert cache.length == 2
        with pytest.raises(ValueError, match='dtype'):
            model.double()(ids, cache=cache)
        assert cache.length == 2


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_cached_logits_low_precision(dtype):
    torch.manual_seed(2)
    model = model_for_test().to(dtype)
    ids = torch.randint(0, 43, (1, 8))
    with torch.inference_mode():
        full = model(ids)
        cache = KVCache(2)
        first = model(ids[:, :5], cache=cache)
        tail = model(ids[:, 5:], cache=cache)
    actual = torch.cat([first, tail], dim=1)
    assert torch.isfinite(actual).all()
    tolerance = .004 if dtype == torch.bfloat16 else .0005
    torch.testing.assert_close(actual, full, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('length', [128, 512, 1024, 2048])
def test_long_prompt_offsets_match_full_forward(length):
    torch.manual_seed(8)
    model = model_for_test(context=2304, layers=1)
    ids = torch.randint(0, 43, (1, length + 2))
    cache = KVCache(1)
    with torch.inference_mode():
        model(ids[:, :length], cache=cache)
        actual = model(ids[:, length:], cache=cache)
        expected = model(ids)[:, length:]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_layer_cache_append_and_consistency_validation():
    cache = LayerKVCache()
    k, v = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
    with torch.no_grad():
        cache.append(k, v)
        result_k, result_v = cache.append(k[:, :, :1], v[:, :, :1])
        torch.testing.assert_close(result_k[:, :, :3], k)
        torch.testing.assert_close(result_v[:, :, :3], v)
        assert cache.length == 4
        with pytest.raises(ValueError, match='shapes'):
            cache.append(k, v[:, :, :1])
    with pytest.raises(ValueError, match='inference-only'):
        cache.append(k, v)
    multi = KVCache(2)
    multi.layers[0] = cache
    with pytest.raises(ValueError, match='inconsistent'):
        _ = multi.length


def test_cached_eos_and_context_stop():
    model = model_for_test(context=8)
    calls = []
    def eos_logits(module, args, output):
        calls.append(args[0].shape[1])
        result = torch.full_like(output, -100.)
        result[..., 4] = 100.
        return result
    hook = model.register_forward_hook(eos_logits)
    try:
        result = generate(model, torch.tensor([[1, 2]]), max_new_tokens=4,
                          eos_token_id=4, use_cache=True)
        assert result.tolist() == [[1, 2, 4]] and calls == [2]
    finally:
        hook.remove()
    assert generate(model, torch.tensor([[1, 2, 3, 4, 5, 6]]),
                    max_new_tokens=20, use_cache=True).shape == (1, 8)
