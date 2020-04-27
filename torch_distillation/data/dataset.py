#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset class(es) for Knowledge Distillation.
"""
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset, TensorDataset

from .utils import chunk, read_tsv


class LanguageModelingDataset(Dataset):
    """
    Dataset class for Language Model sequences where each sample from the dataset will be retrieved by indexing the list of sequences and their corresponding lengths.
    Adapted (in part) from Hugging Face, Inc. (https://github.com/huggingface/transformers/tree/master/examples/distillation)

    Args:
        path: The path to the file containing the sequences.
        tokenizer: The tokenizer.
        encoding: The encoding used to decode the bytes. (default: 'utf-8')
        do_tokenize: Whether to tokenize the data or not. (default: False)
        min_sequence_len: The minimum sequence length. (default: 12)
        max_sequence_len: The maximum sequence length. (default: 512)
    """

    def __init__(
        self,
        path: Union[str, Path],
        tokenizer: Tokenizer,
        encoding: Optional[str] = 'utf-8',
        do_tokenize: Optional[bool] = False,
        min_sequence_len: Optional[int] = 12,
        max_sequence_len: Optional[int] = 512
    ) -> None:
        self._path = Path(path)
        self._tokenizer = tokenizer
        self.min_sequence_len = min_sequence_len
        self.max_sequence_len = max_sequence_len

        # load the data
        self.sequences = []
        with open(self._path, encoding=encoding) as f:
            for line in f:
                self.sequences.append(line.rstrip('\n'))

        if do_tokenize:  # tokenize the data
            self.sequences = [
                sequence.ids for sequence in self._tokenizer.encode_batch(self.sequences)]
        else:  # split the pre-tokenized data
            self.sequences = [sequence.split() for sequence in self.sequences]

        # convert to numpy arrays
        self.sequences = np.array(
            [np.array(sequence, dtype=np.uint16) for sequence in self.sequences])

        # compute sequence lengths
        self.lengths = np.array([len(sequences)
                                 for sequences in self.sequences])

        # create special_tokens_map
        self.special_tokens_map = dict()
        for param, value in self._tokenizer._parameters.items():
            if param.endswith('_token'):
                self.special_tokens_map[param] = self._tokenizer.token_to_id(
                    value)

        # split any sequences that are too long
        self._split_long_sequences()

        # remove any sequences that are too short
        self._remove_short_sequences()

    def _split_long_sequences(self) -> None:
        """Split any sequences that are too long (length > max_sequence_len).
        """
        idxs = self.lengths > self.max_sequence_len

        if idxs.any():
            cls_idx = self.special_tokens_map['cls_token']
            sep_idx = self.special_tokens_map['sep_token']

            new_sequences = []
            new_lengths = []

            for sequence_, len_ in zip(self.sequences, self.lengths):
                if len_ <= self.max_sequence_len:  # sequence is not too long
                    new_sequences.append(sequence_)
                    new_lengths.append(len_)
                else:  # sequence is too long
                    # split sequence into subsequences of length <= max_sequence_len
                    sub_sequences = []
                    for sub_sequence in chunk(sequence_, self.max_sequence_len - 2):
                        if sub_sequence[0] != cls_idx:
                            sub_sequence = np.insert(sub_sequence, 0, cls_idx)
                        if sub_sequence[-1] != sep_idx:
                            sub_sequence = np.insert(
                                sub_sequence, len(sub_sequence), sep_idx)

                        sub_sequences.append(sub_sequence)

                    new_sequences += sub_sequences
                    new_lengths += [len(sub_sequence)
                                    for sub_sequence in sub_sequences]

            # convert new_sequences and new_lengths to numpy arrays
            self.sequences = np.array(new_sequences)
            self.lengths = np.array(new_lengths)

    def _remove_short_sequences(self) -> None:
        """Remove any sequences that are too short (length < min_sequence_len).
        """
        idxs = self.lengths >= self.min_sequence_len
        self.sequences = self.sequences[idxs]
        self.lengths = self.lengths[idxs]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Get a sample (sequence, length) from the dataset.

        Args:
            idx: The sample index.

        Returns:
            A tuple consisting of the sequence and its length.
        """
        return self.sequences[idx], self.lengths[idx]

    def __len__(self) -> int:
        """Get the length (size) of the dataset.

        Returns:
            The length (size) of the dataset.
        """
        return len(self.lengths)

    def sequences_collate_fn(self, batch: List[Tuple[np.ndarray, int]]) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Pad the sequences in the batch (if applicable) and transform both sequences and lengths into torch.LongTensor.

        Args:
            batch: The batch.

        Returns:
            A tuple consiting of the (padded) sequences and their (unpadded) lengths in torch.LongTensor form.
        """
        # unzip the batch
        sequences, lengths = zip(*batch)

        # pad the sequences
        max_sequence_len_batch = max(lengths)
        pad_idx = self.special_tokens_map['pad_token']
        padded_sequences = np.full(
            (len(sequences), max_sequence_len_batch), pad_idx)
        for i, sequence in enumerate(sequences):
            padded_sequences[i, :len(sequence)] = sequence

        # convert to torch.LongTensors
        return torch.LongTensor(padded_sequences), torch.LongTensor(lengths)


GLUE_TASKS_MAPPING = {
    'CoLA': {
        'idxs': {
            'sequence_a': 3,
            'sequence_b': None,
            'label': 1
        }, 'labels': {
            '0': 0,
            '1': 1
        }, 'type': 'classification'
    }, 'MNLI': {
        'idxs': {
            'sequence_a': 8,
            'sequence_b': 9,
            'label': -1
        }, 'labels': {
            'contradiction': 0,
            'entailment': 1,
            'neutral': 2
        }, 'type': 'classification'
    }, 'MRPC': {
        'idxs': {
            'sequence_a': 3,
            'sequence_b': 4,
            'label': 0
        }, 'labels': {
            '0': 0,
            '1': 1
        }, 'type': 'classification'
    }, 'SST-2': {
        'idxs': {
            'sequence_a': 0,
            'sequence_b': None,
            'label': 1
        }, 'labels': {
            '0': 0,
            '1': 1
        }, 'type': 'classification'
    }, 'STS-B': {
        'idxs': {
            'sequence_a': 7,
            'sequence_b': 8,
            'label': -1
        }, 'labels': [None],
        'type': 'regression'
    }, 'QQP': {
        'idxs': {
            'sequence_a': 3,
            'sequence_b': 4,
            'label': 5
        }, 'labels': {
            '0': 0,
            '1': 1
        }, 'type': 'classification'
    }, 'QNLI': {
        'idxs': {
            'sequence_a': 1,
            'sequence_b': 2,
            'label': -1
        }, 'labels': {
            'entailment': 0,
            'not_entailment': 1
        }, 'type': 'classification'
    }, 'RTE': {
        'idxs': {
            'sequence_a': 1,
            'sequence_b': 2,
            'label': -1
        }, 'labels': {
            'entailment': 0,
            'not_entailment': 1
        }, 'type': 'classification'
    }, 'WNLI': {
        'idxs': {
            'sequence_a': 1,
            'sequence_b': 2,
            'label': -1
        }, 'labels': {
            '0': 0,
            '1': 1
        }, 'type': 'classification'
    }
}
GLUE_TASKS = [*GLUE_TASKS_MAPPING]


class GLUETaskDataset(TensorDataset):
    """
    Dataset class for GLUE tasks. 
    Adapted (in part) from Hugging Face, Inc. (https://github.com/huggingface/transformers/tree/master/examples/run_glue.py)

    Args:
        task: The GLUE task.
        glue_dir: The path to the directory containing the GLUE tasks data.
        split: The data split (train, dev or test) to load.
        tokenizer: The tokenizer.
        encoding: The encoding used to decode the bytes. (default: 'utf-8')
        quotechar: The character used to quote special characters. (default: None)
        overwrite_cache: Whether to overwrite the cache (if it exists) or not. (default: False)
    """

    def __init__(
        self,
        task: str,
        glue_dir: Path,
        split: str,
        tokenizer: Tokenizer,
        encoding: Optional[str] = 'utf-8',
        quotchar: Optional[str] = None,
        overwrite_cache: Optional[bool] = False
    ) -> None:
        self.task = task
        self._glue_dir = glue_dir
        self.split = split
        self._tokenizer = tokenizer

        # hot fix for MNLI-MM task_dir
        if self.task == 'MNLI-MM':
            self._task_dir = self._glue_dir / 'MNLI'
        else:
            self._task_dir = self._glue_dir / self.task

        # hot fix for MNLI and MNLI-MM dev data
        if self.split == 'dev':
            if self.task == 'MNLI':
                self.split += '_matched'
            elif self.task == 'MNLI-MM':
                self.split += '_mismatched'            

        # get the path
        self._path = self._task_dir / f'{self.split}.tsv'
        self._cached_sequences_path = self._task_dir / f'cached_{self.split}_{self._tokenizer.__class__.__name__}_sequences.pth'
        self._cached_attention_masks_path = self._task_dir / f'cached_{self.split}_{self._tokenizer.__class__.__name__}_attention_masks.pth'
        self._cached_labels_path = self._task_dir / f'cached_{self.split}_{self._tokenizer.__class__.__name__}_labels.pth'

        # get the mapping
        self.mapping = GLUE_TASKS_MAPPING[self.task]

        if (
            self._cached_sequences_path.exists()
            and self._cached_attention_masks_path.exists()
            and self._cached_labels_path.exists()
            and not overwrite_cache
        ):
            # load tensors from cache
            self.sequences = torch.load(self._cached_sequences_path)
            self.attention_masks = torch.load(self._cached_attention_masks_path)
            self.labels = torch.load(self._cached_labels_path)
        else:
            if not self._path.exists():
                raise FileNotFoundError(f'{self.task} GLUE task {self.split} data file not found: {str(self._path)}')

            # load the data
            data = read_tsv(self._path, encoding=encoding, quotechar=quotchar)
            if self.task != 'CoLA':
                data = data[1:]

            # convert data to sequences (pairs) and labels
            self.sequences = []
            self.labels = []
            for line in data:
                try:
                    # first sequence
                    sequence = line[self.mapping['idxs']['sequence_a']]

                    # second sequence (optional)
                    if self.mapping['idxs']['sequence_b'] is not None:
                        sequence = (sequence, line[self.mapping['idxs']['sequence_b']])

                    # label
                    label = line[self.mapping['idxs']['label']]
                    if self.mapping['type'] == 'classification':
                        label = self.mapping['labels'][label]
                    elif self.mapping['type'] == 'regression':
                        label = float(label)

                    self.sequences.append(sequence)
                    self.labels.append(label)
                except IndexError:
                    continue

            # tokenize the sequences (pairs)
            encoded_sequences = self._tokenizer.encode_batch(self.sequences)
            self.sequences = [encoded_sequence.ids for encoded_sequence in encoded_sequences]
            self.attention_masks = [encoded_sequence.attention_mask for encoded_sequence in encoded_sequences]

            # convert to torch tensors
            self.sequences = torch.tensor(self.sequences, dtype=torch.long)
            self.attention_masks = torch.tensor(self.attention_masks, dtype=torch.long)
            if self.mapping['type'] == 'classification':
                self.labels = torch.tensor(self.labels, dtype=torch.long)
            else:
                self.labels = torch.tensor(self.labels, dtype=torch.float)

            if not self.sequences.size(0) == self.attention_masks.size(0) == self.labels.size(0):
                raise ValueError('The sequences, attention masks and labels are of unequal size.')

            # save tensors to cache
            torch.save(self.sequences, self._cached_sequences_path)
            torch.save(self.attention_masks, self._cached_attention_masks_path)
            torch.save(self.labels, self._cached_labels_path)

        self.tensors = (self.sequences, self.attention_masks, self.labels)
