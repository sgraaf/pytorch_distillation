#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Loss classes for Knowledge Distillation.
"""
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class SoftTargetLoss(_Loss):
    r"""The soft target loss from the "Distilling the Knowledge in a Neural 
    Network" paper by Hinton et al. (2015)

    Args:
        temperature: The softmax temperature. (default: 1.0)
        reduction: The reduction to apply to the output. (default: 'mean')
    """

    def __init__(
        self,
        temperature: Optional[float] = 1.0,
        reduction: Optional[str] = 'mean',
    ) -> None:
        super(SoftTargetLoss, self).__init__(reduction)
        self.temperature = temperature

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.kl_div(
            input=F.log_softmax(input / self.temperature, dim=-1),
            target=F.softmax(target / self.temperature, dim=-1),
            reduction=self.reduction
        )


class HintonLoss(_Loss):
    r"""The distillation loss from the "Distilling the Knowledge in a Neural 
    Network" paper by Hinton et al. (2015)

    Args:
        alpha: The relative weight of the soft target loss. (default: 0.5)
        temperature: The softmax temperature. (default: 2.0)
        reduction: The reduction to apply to the output. (default: 'mean')
    """

    def __init__(
        self,
        alpha: Optional[float] = 0.5,
        temperature: Optional[float] = 2.0,
        reduction: Optional[Union[str, Sequence[str]]] = 'mean',
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f'The alpha value must be in range [0.0, 1.0], but got {alpha}')
        if isinstance(reduction, str):
            soft_target_reduction = hard_target_reduction = reduction
        if isinstance(reduction, Sequence):
            if len(reduction) != 2:
                raise ValueError(
                    f'2 reduction values expected, but got {len(reduction)} value(s)')
            soft_target_reduction, hard_target_reduction = reduction

        super(HintonLoss, self).__init__(reduction)
        self.alpha = alpha
        self.temperature = temperature
        self.soft_target_loss = SoftTargetLoss(
            temperature=self.temperature,
            reduction=soft_target_reduction
        )
        self.hard_target_loss = nn.CrossEntropyLoss(
            reduction=hard_target_reduction)

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        return self.alpha * self.soft_target_loss(input1, input2) * self.temperature ** 2 + \
            (1.0 - self.alpha) * self.hard_target_loss(input1, target)


class SanhLoss(_Loss):
    r"""The distillation loss from the "DistilBERT, a distilled version of BERT:
     smaller, faster, cheaper and lighter" paper by Sanh et al. (2019)

    Args:
        alphas: The relative weights of the losses. (default: (0.33, 0.33, 0.33))
        temperature: The softmax temperature. (default: 2.0)
        reduction: The reduction to apply to the output. (default: 'mean')
    """

    def __init__(
        self,
        alphas: Optional[Sequence[float]] = (0.33, 0.33, 0.33),
        temperature: Optional[float] = 2.0,
        reduction: Optional[Union[str, Sequence[str]]] = 'mean',
    ) -> None:
        if len(alphas) != 3:
            raise ValueError(
                f'3 alpha values expected, but got {len(alphas)} alpha values')
        for alpha in alphas:
            if not alpha >= 0.0:
                raise ValueError(
                    f'All alpha values must be strictly non-negative, but got {alpha}')
        if not sum(alphas) > 0.0:
            raise ValueError(
                f'The sum of all alpha values must be strictly positive, but got {sum(alphas)}')

        if isinstance(reduction, str):
            soft_target_reduction = hard_target_reduction = cosine_emb_reduction = reduction
        if isinstance(reduction, Sequence):
            if len(reduction) != 3:
                raise ValueError(
                    f'3 reduction values expected, but got {len(reduction)} value(s)')
            soft_target_reduction, hard_target_reduction, cosine_emb_reduction = reduction

        super(SanhLoss, self).__init__(reduction)
        self.soft_target_alpha, self.hard_target_alpha, self.cosine_emb_alpha = alphas
        self.temperature = temperature
        self.soft_target_loss = SoftTargetLoss(
            temperature=self.temperature,
            reduction=soft_target_reduction
        )
        self.hard_target_loss = nn.CrossEntropyLoss(
            ignore_index=-100,
            reduction=hard_target_reduction
        )
        self.cosine_emb_loss = nn.CosineEmbeddingLoss(
            reduction=cosine_emb_reduction
        )

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        input3: torch.Tensor,
        input4: torch.Tensor,
        input5: torch.Tensor,
        target1: torch.Tensor,
        target2: torch.Tensor,
    ) -> torch.Tensor:
        return self.soft_target_alpha * self.soft_target_loss(input1, input2) * self.temperature ** 2 + \
            self.hard_target_alpha * self.hard_target_loss(input3, target1) + \
            self.cosine_emb_alpha * self.cosine_emb_loss(input4, input5, target2)
