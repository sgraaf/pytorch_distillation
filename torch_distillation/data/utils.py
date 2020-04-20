#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data-related utility functions for Knowledge Distillation.
"""
import bisect
import copy
import csv
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple


def chunk(l: Sequence[Any], n: int) -> Tuple[Sequence[Any]]:
    """Divide a sequence l into chunks of size n.

    Args:
        l: The sequence to chunk.
        n: The size of the chunk(s).

    Returns:
        A tuple consisting of the sequence chunks.
    """
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


def quantize(l: Sequence[int], bins: List[int]) -> List[int]:
    """Quantize the values of a sequence, given the bins.
    
    Args:
        l: The sequence to quantize.
        bins: The bins.
    
    Returns:
        A list of quantized values.
    """
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    quantized_l = list(map(lambda y: bisect.bisect_right(bins, y), l))

    return quantized_l

def read_csv(file: Path, encoding: Optional[str] = 'utf-8', delimiter: Optional[str] = ',', quotechar: Optional[str] = None):
    """Reads a comma-separated value (CSV) file.
    
    Args:
        file: The path to the CSV-file.
        encoding: The encoding used to decode the bytes.
        delimiter: The character used to seperate the fields.
        quotechar: The character used to quote fields containing special characters.
    """
    with open(file, 'r', encoding=encoding) as f:
        return list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))

def read_tsv(file: Path, encoding: Optional[str] = 'utf-8', quotechar: Optional[str] = None):
    """Reads a tab-separated value (TSV) file.
    
    Args:
        file: The path to the TSV-file.
        encoding: The encoding used to decode the bytes.
        quotechar: The character used to quote fields containing special characters.
    """
    return read_csv(file, encoding=encoding, delimiter='\t', quotechar=quotechar)
