#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Distiller classes for Knowledge Distillation.
"""
from logging import Logger
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


class _Distiller(object):

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        dataloader: DataLoader,
        loss_fn: Loss,
        optimizer: Optimizer,
        num_epochs: int,
        scheduler: Optional[Scheduler] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.logger = logger

    def distill(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.teacher.__class__.__name__}->{self.student.__class__.__name__})'


class Distiller(_Distiller):
    """Distiller class for Knowledge Distillation.
    
    Args:
        student: The student model.
        teacher: The teacher model.
        dataloader: The dataloader from which to load the batches.
        loss_fn: The loss function.
        optimizer: The optimizer.
        num_epochs: The number of distillation epochs (iterations over the complete dataset).
        scheduler: The learning rate scheduler. (default: None)
        use_cuda: Whether to use cuda or not. (default: False)
        logger: The logger. (default: None)
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        dataloader: DataLoader,
        loss_fn: Loss,
        optimizer: Optimizer,
        num_epochs: int,
        scheduler: Optional[Scheduler] = None,
        use_cuda: Optional[bool] = False,
        logger: Optional[Logger] = None,
    ) -> None:
        super(Distiller, self).__init__(
            student,
            teacher,
            dataloader,
            loss_fn,
            optimizer,
            num_epochs,
            scheduler,
            logger,
        )
        self.use_cuda = use_cuda

        if self.use_cuda:  # send models to GPU
            self.student = self.student.to('cuda')
            self.teacher = self.teacher.to('cuda')

    def distill(self):
        # put student in train mode and teacher in eval mode
        self.student.train()
        self.teacher.eval()

        for epoch in range(self.num_epochs):
            if self.logger is not None:
                self.logger.info(f'Starting with epoch {epoch+1}/{self.num_epochs}')
            
            with tqdm(self.dataloader, desc='Distilling', total=len(self.dataloader)) as pbar:
                for step, batch in enumerate(pbar):
                    # clear all gradients
                    self.optimizer.zero_grad()

                    # unpack batch
                    input, target = batch

                    # send input to device
                    input = input.to('cuda')

                    # forward pass
                    student_logits = self.student(input)
                    with torch.no_grad():
                        teacher_logits = self.teacher(input)

                    # compute the loss
                    loss = self.loss_fn(student_logits, teacher_logits)

                    # update the last_loss in the progress bar
                    pbar.set_postfix({'last_loss': loss.item()})

                    # backward pass
                    loss.backward()

                    # update the parameters
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()


class DistributedDistiller(_Distiller):
    """DistributedDistiller class for distributed (multi-GPU) Knowledge Distillation.
    
    Args:
        student: The student model.
        teacher: The teacher model.
        dataloader: The dataloader from which to load the batches.
        loss_fn: The loss function.
        optimizer: The optimizer.
        num_epochs: The number of distillation epochs (iterations over the complete dataset).
        local_rank: The local rank of the process.
        scheduler: The learning rate scheduler. (default: None)
        logger: The logger. (default: None)
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        dataloader: DataLoader,
        loss_fn: Loss,
        optimizer: Optimizer,
        num_epochs: int,
        local_rank: int,
        scheduler: Scheduler = None,
        logger: Logger = None,
    ) -> None:
        super(DistributedDistiller, self).__init__(
            student,
            teacher,
            dataloader,
            loss_fn,
            optimizer,
            num_epochs,
            scheduler,
            logger,
        )
        self.local_rank = local_rank

        # send models to GPU
        self.student = self.student.to(f'cuda:{self.local_rank}')
        self.teacher = self.teacher.to(f'cuda:{self.local_rank}')

        # initialize distributed data parallel (DDP)
        self.student = DDP(
            self.student,
            device_ids=[self.local_rank],
            output_device=self.local_rank
        )

    def distill(self):
        # put student in train mode and teacher in eval mode
        self.student.train()
        self.teacher.eval()

        for epoch in range(self.num_epochs):
            # synchronize all processes
            torch.distributed.barrier()

            if self.logger is not None:
                self.logger.info(f'Starting with epoch {epoch+1}/{self.num_epochs}')

            with tqdm(self.dataloader, desc='Distilling', total=len(self.dataloader)) as pbar:
                for step, batch in enumerate(pbar):
                    # clear all gradients
                    self.optimizer.zero_grad()

                    # unpack batch
                    input, target = batch

                    # send input to device
                    input = input.to(f'cuda:{self.local_rank}')

                    # forward pass
                    student_logits = self.student(input)
                    with torch.no_grad():
                        teacher_logits = self.teacher(input)

                    # compute the loss
                    loss = self.loss_fn(student_logits, teacher_logits)
                    loss = loss.mean()
                    
                    # update the last_loss in the progress bar
                    pbar.set_postfix({'last_loss': loss.item()})

                    # backward pass
                    loss.backward()

                    # update the parameters
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
