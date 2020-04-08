#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sampler class(es) for Knowledge Distillation. 
"""
from collections import defaultdict
from typing import List, Optional

from torch.utils.data.sampler import Sampler


class GroupedBatchSampler(Sampler):
    """GroupedBatchSampler class for grouped Language Model sequences sampling.
    Adapted (in part) from Hugging Face, Inc. (https://github.com/huggingface/transformers/tree/master/examples/distillation)

    Args:
        sampler: The base sampler.
        group_idxs: The list of group indices (one group index per sample).
        batch_size: The batch size.
        drop_last: Whether to drop the last batch if its size would be less than batch_size. (default: False)
    """

    def __init__(
        self,
        sampler: Sampler,
        group_idxs: List[int],
        batch_size: int,
        drop_last: Optional[bool] = False,
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise ValueError(
                f'sampler should be an instance of torch.utils.data.Sampler, but got sampler={sampler}')
        if batch_size <= 0:
            raise ValueError(
                f'batch_size should be a positive integer value, but got batch_size={batch_size}')
        self.sampler = sampler
        self.group_idxs = group_idxs
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        group_batches = defaultdict(list)
        n_batches = 0

        for example_idx in self.sampler:
            # get the group index of the example
            group_idx = self.group_idxs[example_idx]

            # add the example to the batch of that group
            group_batches[group_idx].append(example_idx)

            if len(group_batches[group_idx]) == self.batch_size:  # full batch
                yield group_batches[group_idx]
                n_batches += 1
                del group_batches[group_idx]

        # yield the remaining batches that do not satisfy the group criteria
        n_batches_remaining = len(self) - n_batches
        if n_batches_remaining > 0:
            # group the remaining batches by similar sequence lengths
            batch = []
            for _, example_idxs in sorted(group_batches.items(), key=lambda x: x[0]):
                batch += example_idxs
                if len(batch) >= self.batch_size:  # (over)full batch
                    yield batch[:self.batch_size]
                    batch = batch[self.batch_size:]
                    n_batches_remaining -= 1

            # yield the last (incomplete) batch
            if len(batch) > 0 and not self.drop_last:
                yield batch

    def __len__(self) -> int:
        """Return the number of batches.
        """
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
