"""Deterministic, resumable ordering of indexable CPU micro-batches."""

from collections.abc import Sequence

import torch


class ResumableBatchStream:
    """No worker prefetch: cursor advances only when Trainer consumes a batch.

    fingerprint must identify immutable data plus batching/collation settings.
    A future shard/JSONL dataset can implement the same Sequence interface.
    """

    def __init__(self, batches: Sequence, *, fingerprint: str, seed: int = 42,
                 shuffle: bool = True) -> None:
        if len(batches) == 0 or not fingerprint:
            raise ValueError("nonempty data and data fingerprint are required")
        self.batches, self.fingerprint = batches, fingerprint
        self.seed, self.shuffle = seed, shuffle
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch, self.cursor = 0, 0
        self.order = self._new_order()

    def _new_order(self):
        return torch.randperm(len(self.batches), generator=self.generator) if self.shuffle else torch.arange(len(self.batches))

    def __iter__(self):
        return self

    def __next__(self):
        if self.cursor >= len(self.order):
            raise StopIteration
        batch = self.batches[int(self.order[self.cursor])]
        self.cursor += 1
        return batch

    def next_epoch(self) -> None:
        if self.cursor != len(self.order):
            raise ValueError("cannot advance epoch before consuming all batches")
        self.epoch += 1
        self.cursor = 0
        self.order = self._new_order()

    def state_dict(self) -> dict:
        return dict(fingerprint=self.fingerprint, size=len(self.batches), seed=self.seed,
                    shuffle=self.shuffle, epoch=self.epoch, cursor=self.cursor,
                    order=self.order.clone(), generator_state=self.generator.get_state())

    def validate_state(self, state: dict) -> None:
        for key, expected in [('fingerprint', self.fingerprint), ('size', len(self.batches)),
                              ('seed', self.seed), ('shuffle', self.shuffle)]:
            if state[key] != expected:
                raise ValueError(f"data stream {key} mismatch")
        order = state['order']
        if (order.dtype != torch.int64 or order.shape != (len(self.batches),)
                or not torch.equal(order.sort().values, torch.arange(len(self.batches)))
                or not 0 <= state['cursor'] <= len(self.batches) or state['epoch'] < 0):
            raise ValueError("invalid data cursor/order")
        torch.Generator().set_state(state['generator_state'])

    def load_state_dict(self, state: dict) -> None:
        self.validate_state(state)
        self.epoch, self.cursor = state['epoch'], state['cursor']
        self.order = state['order'].clone()
        self.generator.set_state(state['generator_state'])
