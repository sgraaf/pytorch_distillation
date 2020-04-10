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
from torch.nn.utils import clip_grad_norm_
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
        num_gradient_accumulation_steps: Optional[int] = 1,
        max_gradient_norm: Optional[float] = None,
        scheduler: Optional[Scheduler] = None,
        use_tqdm: Optional[bool] = True,
        logger: Optional[Logger] = None,

    ) -> None:
        self.student = student
        self.teacher = teacher
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.num_gradient_accumulation_steps = num_gradient_accumulation_steps
        self.max_gradient_norm = max_gradient_norm
        self.scheduler = scheduler
        self.use_tqdm = use_tqdm
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
        use_cuda: Whether to use cuda or not. (default: False)
        num_gradient_accumulation_steps: The number of steps in which the gradient are accumulated (default: 1)
        max_gradient_norm: The maximum gradient norm (for clipping). (default: None)
        scheduler: The learning rate scheduler. (default: None)
        use_tqdm: Whether to use tqdm (progress bar) or not. (default: True)
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
        use_cuda: Optional[bool] = False,
        num_gradient_accumulation_steps: Optional[int] = 1,
        max_gradient_norm: Optional[float] = None,
        scheduler: Optional[Scheduler] = None,
        use_tqdm: Optional[bool] = True,
        logger: Optional[Logger] = None,
    ) -> None:
        super(Distiller, self).__init__(
            student,
            teacher,
            dataloader,
            loss_fn,
            optimizer,
            num_epochs,
            num_gradient_accumulation_steps,
            max_gradient_norm,
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

            # initialize the progress bar
            if self.use_tqdm:
                pbar = tqdm(
                    desc=f'Distilling [epoch {epoch+1}/{self.num_epochs}]',
                    total=len(self.dataloader),
                    unit='batch',
                    leave=False
                )
            
            for step, batch in enumerate(self.dataloader):
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

                # rescale the loss
                loss /= self.num_gradient_accumulation_steps

                # backward pass
                loss.backward()

                if step % self.num_gradient_accumulation_steps == 0:
                    # clip the gradient
                    if self.max_gradient_norm is not None:
                        clip_grad_norm_(self.student.parameters(), self.max_gradient_norm)
                    
                    # update the parameters
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                    # clear all gradients
                    self.optimizer.zero_grad()

                # update the progress bar
                if self.use_tqdm:
                    pbar.update()
                    pbar.set_postfix({'last_loss': loss.item()})
            
            # close the progress bar
            if self.use_tqdm:
                pbar.close()
                    


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
        is_master: Whether the local process is the master process. (default: True)
        num_gradient_accumulation_steps: The number of steps in which the gradient are accumulated (default: 1)
        max_gradient_norm: The maximum gradient norm (for clipping). (default: None)
        scheduler: The learning rate scheduler. (default: None)
        use_tqdm: Whether to use tqdm (progress bar) or not. (default: True)
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
        is_master: Optional[bool] = True,
        num_gradient_accumulation_steps: Optional[int] = 1,
        max_gradient_norm: Optional[float] = None,
        scheduler: Optional[Scheduler] = None,
        use_tqdm: Optional[bool] = True,
        logger: Optional[Logger] = None,
    ) -> None:
        super(DistributedDistiller, self).__init__(
            student,
            teacher,
            dataloader,
            loss_fn,
            optimizer,
            num_epochs,
            num_gradient_accumulation_steps,
            max_gradient_norm,
            scheduler,
            logger,
        )
        self.local_rank = local_rank
        self.is_master = is_master

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

            if self.is_master and self.logger is not None:
                self.logger.info(f'Starting with epoch {epoch+1}/{self.num_epochs}')

            # initialize the progress bar
            if self.use_tqdm:
                pbar = tqdm(
                    desc=f'Distilling [epoch {epoch+1}/{self.num_epochs}]',
                    total=len(self.dataloader),
                    unit='batch',
                    leave=False
                )
            
            for step, batch in enumerate(self.dataloader):
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
                
                # rescale the loss
                loss /= self.num_gradient_accumulation_steps

                # backward pass
                loss.backward()

                if step % self.num_gradient_accumulation_steps == 0:
                    # clip the gradient
                    if self.max_gradient_norm is not None:
                        clip_grad_norm_(self.student.parameters(), self.max_gradient_norm)

                    # update the parameters
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                    # clear all gradients
                    self.optimizer.zero_grad()

                # update the progress bar
                if self.use_tqdm:
                    pbar.update()
                    pbar.set_postfix({'last_loss': loss.item()})
            
            # close the progress bar
            if self.use_tqdm:
                pbar.close()
