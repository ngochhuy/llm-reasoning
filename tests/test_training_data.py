import json
import sys
from array import array

import pytest
import torch

from llm_reasoning.data_stream import ResumableBatchStream
from llm_reasoning.training_data import PretrainBatches, SFTBatches


def test_pretrain_blocks_and_resumable_cursor(tmp_path):
    root = tmp_path / 'pretrain'
    shard = root / 'fineweb' / 'train' / 'shard_00000.bin'
    shard.parent.mkdir(parents=True)
    with shard.open('wb') as file:
        array('H', range(12)).tofile(file)
    (root / 'manifest.json').write_text(json.dumps(dict(token_dtype='uint16', byte_order=sys.byteorder,
        block_size=3, sources={'fineweb': {'shards': {'train': {'files': [shard.name]}}}})))
    batches = PretrainBatches(root, 'train', batch_size=2)
    assert len(batches) == 2
    assert batches[0]['input_ids'].tolist() == [[0, 1, 2], [3, 4, 5]]
    stream = ResumableBatchStream(batches, fingerprint=batches.fingerprint, shuffle=False)
    assert next(stream)['input_ids'].shape == (2, 3)
    restored = ResumableBatchStream(batches, fingerprint=batches.fingerprint, shuffle=False)
    restored.load_state_dict(stream.state_dict())
    assert next(restored)['input_ids'].tolist() == [[6, 7, 8], [9, 10, 11]]
    with shard.open('ab') as file:
        array('H', [12, 13, 14]).tofile(file)
    assert PretrainBatches(root, 'train', batch_size=2).fingerprint != batches.fingerprint


def test_sft_jsonl_mask_and_fingerprint(tmp_path):
    root = tmp_path / 'sft'
    root.mkdir()
    (root/'manifest.json').write_text(json.dumps({'sft_sources': ['gsm8k']}))
    rows = [{'input_ids': [1, 2, 3], 'loss_mask': [0, 1, 1]},
            {'input_ids': [4, 5], 'loss_mask': [0, 1]},
            {'input_ids': [6, 7, 8, 9], 'loss_mask': [0, 0, 1, 1]}]
    (root/'gsm_train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    batches = SFTBatches(root, 'train', batch_size=2, pad_token_id=99, max_seq_len=4)
    assert len(batches) == 2
    first = batches[0]
    assert first['input_ids'].tolist() == [[1, 2, 3], [4, 5, 99]]
    assert first['loss_mask'].tolist() == [[False, True, True], [False, True, False]]
    assert batches[1]['input_ids'].shape == (1, 4)
    with pytest.raises(IndexError):
        _ = batches[2]
    with pytest.raises(ValueError, match='length'):
        _ = SFTBatches(root, 'train', batch_size=2, pad_token_id=99, max_seq_len=3)[1]
    with (root/'gsm_train.jsonl').open('a') as file:
        file.write(json.dumps(rows[0])+'\n')
    assert SFTBatches(root, 'train', batch_size=2, pad_token_id=99, max_seq_len=4).fingerprint != batches.fingerprint
