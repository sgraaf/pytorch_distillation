#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data-related utility functions for Knowledge Distillation.
"""
import bisect
import copy
from typing import Any, Iterable, List, Sequence, Tuple


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
