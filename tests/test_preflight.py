import json
import sys
from array import array

from llm_reasoning.checkpoint import file_sha256
from llm_reasoning.config import TokenizerConfig
from llm_reasoning.preflight import inspect
from llm_reasoning.tokenizer import ByteBPETokenizer


def statuses(result):
    return {item['name']: item['status'] for item in result['checks']}


def test_missing_artifacts_fail_explicitly(tmp_path):
    result = inspect(tmp_path/'missing', tmp_path/'out', tmp_path/'not_a_repo', production=False)
    assert result['ready'] is False
    assert statuses(result)['data_manifests'] == 'fail'


def test_dry_run_data_rejected_without_losing_valid_evidence(tmp_path):
    data = tmp_path/'data'
    pretrain = data/'pretrain'
    sft = data/'sft_gsm8k'
    pretrain.mkdir(parents=True)
    sft.mkdir()
    tokenizer_path = pretrain/'tokenizer.json'
    ByteBPETokenizer(TokenizerConfig(vocab_size=262)).save(tokenizer_path)
    sources = {}
    for source in ('fineweb', 'math'):
        shard = pretrain/source/'train'/'shard_00000.bin'
        shard.parent.mkdir(parents=True)
        with shard.open('wb') as file:
            array('H', range(512)).tofile(file)
        sources[source] = {'shards': {'train': {'files': [shard.name], 'written_tokens': 512},
                                     'validation': {'files': [], 'written_tokens': 0}}}
    sha = file_sha256(tokenizer_path)
    (pretrain/'manifest.json').write_text(json.dumps(dict(tokenizer_sha256=sha, vocab_size=262,
        block_size=512, token_dtype='uint16', byte_order=sys.byteorder, sources=sources)))
    (sft/'manifest.json').write_text(json.dumps(dict(tokenizer_sha256=sha,
        sft_sources=['gsm8k'], gsm_split={'validation': 500, 'test': 1319}, test_examples=1319,
        splits={'gsm_train': {'accepted': 1}, 'gsm_validation': {'accepted': 1}})))
    for name in ('gsm_train.jsonl', 'gsm_validation.jsonl', 'gsm_test.jsonl'):
        (sft/name).write_text('{}\n')
    result = inspect(data, tmp_path/'output', tmp_path/'not_a_repo', production=False)
    found = statuses(result)
    assert found['tokenizer_hash'] == found['pretrain_format'] == 'pass'
    assert found['fineweb_train_shards'] == found['math_train_shards'] == 'fail'
    assert found['fineweb_validation_shards'] == found['math_validation_shards'] == 'fail'
    assert found['sft_train_rows'] == 'pass'
    assert result['ready'] is False

    for path in sft.iterdir():
        path.unlink()
    sft.rmdir()
    pretrain_only = inspect(data, tmp_path/'output', tmp_path/'not_a_repo',
                            production=False, require_sft=False)
    found = statuses(pretrain_only)
    assert found['tokenizer_hash'] == found['pretrain_format'] == 'pass'
    assert not any(name.startswith('sft_') for name in found)
    assert statuses(inspect(data, tmp_path/'output', tmp_path/'not_a_repo',
                            production=False))['data_manifests'] == 'fail'
