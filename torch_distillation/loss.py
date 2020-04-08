#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Loss class(es) for Knowledge Distillation.
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class KDLoss(_Loss):
    """The Knowledge Distillation (KD) loss.

    Args:
        temperature: The softmax temperature. (default: 1.0)
        reduction: The reduction to apply to the output. (default: 'mean')
    """

    def __init__(
        self,
        temperature: Optional[float] = 1.0,
        reduction: Optional[str] = 'mean',
    ) -> None:
        super(KDLoss, self).__init__(reduction)
        self.temperature = temperature

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.kl_div(
            input=F.log_softmax(input / self.temperature, dim=-1),
            target=F.softmax(target / self.temperature, dim=-1),
            reduction=self.reduction
        ) * self.temperature ** 2
