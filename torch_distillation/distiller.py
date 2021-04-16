#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Distiller classes for Knowledge Distillation.
"""
import math
from logging import Logger
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import _Loss as Loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import HintonLoss, SanhLoss


class Distiller(object):
    r"""Base distiller.

    Args:
        student: The student model.
        teacher: The teacher model.
        dataloader: The dataloader from which to load the batches.
        loss_fn: The loss function. (default: None)
        optimizer: The optimizer. (default: None)
        lr_scheduler: The learning rate lr_scheduler. (default: None)
        num_epochs: The number of distillation epochs (iterations over the complete dataset).
        num_gradient_accumulation_steps: The number of steps in which the gradient are accumulated (default: 1)
        max_gradient_norm: The maximum gradient norm (for clipping). (default: None)
        use_cuda: Whether to use cuda or not. (default: False)
        local_rank: The local rank of the process (default: 0)
        use_distributed: Whether to use distributed training (distillation) or not. (default: False)
        is_master: Whether the current process is the master process. (default: True)
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
        lr_scheduler: Optional[LRScheduler] = None,
        num_epochs: Optional[int] = 1,
        num_gradient_accumulation_steps: Optional[int] = 1,
        max_gradient_norm: Optional[float] = None,
        use_cuda: Optional[bool] = False,
        local_rank: Optional[int] = 0,
        use_distributed: Optional[bool] = False,
        is_master: Optional[bool] = True,
        use_tqdm: Optional[bool] = True,
        logger: Optional[Logger] = None,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_epochs = num_epochs
        self.num_gradient_accumulation_steps = num_gradient_accumulation_steps
        self.max_gradient_norm = max_gradient_norm
        self.use_cuda = use_cuda
        self.local_rank = local_rank
        self.use_distributed = use_distributed
        self.is_master = is_master
        self.use_tqdm = use_tqdm
        self.logger = logger

        # send models to GPU
        if self.use_cuda:
            self.student = self.student.to(f'cuda:{self.local_rank}')
            self.teacher = self.teacher.to(f'cuda:{self.local_rank}')

        # initialize distributed data parallel (DDP)
        if self.use_distributed:
            self.student = DDP(
                self.student,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )

    def distill(self):
        # put student in train mode and teacher in eval mode
        self.student.train()
        self.teacher.eval()

        # keep track of the last loss
        self.last_loss = 0

        for epoch in range(self.num_epochs):
            # synchronize all processes
            if self.use_distributed:
                torch.distributed.barrier()

            if self.is_master and self.logger is not None:
                self.logger.info(
                    f'Starting with epoch {epoch+1}/{self.num_epochs}')

            # initialize the progress bar
            if self.use_tqdm:
                pbar = tqdm(
                    desc=f'Distilling [epoch {epoch+1}/{self.num_epochs}]',
                    total=len(self.dataloader),
                    unit='batch',
                    leave=False,
                )

            for step, batch in enumerate(self.dataloader):
                # unpack batch
                input, target = batch

                # send input and target to GPU
                if self.use_cuda:
                    input = input.to(f'cuda:{self.local_rank}')
                    target = target.to(f'cuda:{self.local_rank}')

                # forward pass
                student_logits = self.student(input)
                with torch.no_grad():
                    teacher_logits = self.teacher(input)

                # compute the loss
                loss = self.loss_fn(student_logits, teacher_logits, target)
                self.last_loss = loss.item()

                if self.use_distributed:
                    loss = loss.mean()

                # rescale the loss
                loss /= self.num_gradient_accumulation_steps

                # backward pass
                loss.backward()

                if step % self.num_gradient_accumulation_steps == 0:
                    # clip the gradient
                    if self.max_gradient_norm is not None:
                        clip_grad_norm_(
                            self.student.parameters(),
                            self.max_gradient_norm
                        )

                    # update the parameters
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    # clear all gradients
                    self.optimizer.zero_grad()

                # update the progress bar
                if self.use_tqdm:
                    pbar.update()
                    pbar.set_postfix({'last_loss': self.last_loss})

            # close the progress bar
            if self.use_tqdm:
                pbar.close()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.teacher.__class__.__name__}->{self.student.__class__.__name__})'


class HintonDistiller(Distiller):
    r"""The distiller from the "Distilling the Knowledge in a Neural Network" 
    paper by Hinton et al. (2015)

    Args:
        student: The student model.
        teacher: The teacher model.
        dataloader: The dataloader from which to load the batches.
        loss_fn: The loss function. (default: None)
        optimizer: The optimizer. (default: None)
        lr_scheduler: The learning rate lr_scheduler. (default: None)
        num_epochs: The number of distillation epochs (iterations over the complete dataset).
        num_gradient_accumulation_steps: The number of steps in which the gradient are accumulated (default: 1)
        max_gradient_norm: The maximum gradient norm (for clipping). (default: None)
        use_cuda: Whether to use cuda or not. (default: False)
        local_rank: The local rank of the process (default: 0)
        use_distributed: Whether to use distributed training (distillation) or not. (default: False)
        is_master: Whether the current process is the master process. (default: True)
        use_tqdm: Whether to use tqdm (progress bar) or not. (default: True)
        logger: The logger. (default: None)
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        dataloader: DataLoader,
        loss_fn: Optional[Loss] = None,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        num_epochs: Optional[int] = 1,
        num_gradient_accumulation_steps: Optional[int] = 1,
        max_gradient_norm: Optional[float] = None,
        use_cuda: Optional[bool] = False,
        local_rank: Optional[int] = 0,
        use_distributed: Optional[bool] = False,
        is_master: Optional[bool] = True,
        use_tqdm: Optional[bool] = True,
        logger: Optional[Logger] = None,
    ) -> None:
        super(HintonDistiller, self).__init__(
            student,
            teacher,
            dataloader,
            loss_fn,
            optimizer,
            lr_scheduler,
            num_epochs,
            num_gradient_accumulation_steps,
            max_gradient_norm,
            use_cuda,
            local_rank,
            use_distributed,
            is_master,
            use_tqdm,
            logger,
        )
        if self.loss_fn is None:
            self.loss_fn = HintonLoss()

        if self.optimizer is None:
            self.optimizer = optim.SGD(
                self.student.parameters(),
                lr=0.1,
                momentum=0.9,
                weight_decay=1e-4
            )


class SanhDistiller(Distiller):
    r"""The distiller from the "DistilBERT, a distilled version of BERT:
     smaller, faster, cheaper and lighter" paper by Sanh et al. (2019)

    Args:
        student: The student model.
        teacher: The teacher model.
        dataloader: The dataloader from which to load the batches.
        token_probabilities: The token probabilities.
        pred_proportion: The proportion of tokens to make a prediction for.
        mask_probability: The probability of masking out a token for which a prediction needs to be made.
        keep_probability: The probability of keeping a token for which a prediction needs to be made.
        rand_probability: The probability of randomly replacing a token for which a prediction needs to be made.
        loss_fn: The loss function. (default: None)
        optimizer: The optimizer. (default: None)
        lr_scheduler: The learning rate lr_scheduler. (default: None)
        num_epochs: The number of distillation epochs (iterations over the complete dataset).
        num_gradient_accumulation_steps: The number of steps in which the gradient are accumulated (default: 1)
        max_gradient_norm: The maximum gradient norm (for clipping). (default: None)
        use_cuda: Whether to use cuda or not. (default: False)
        local_rank: The local rank of the process (default: 0)
        use_distributed: Whether to use distributed training (distillation) or not. (default: False)
        is_master: Whether the current process is the master process. (default: True)
        use_tqdm: Whether to use tqdm (progress bar) or not. (default: True)
        logger: The logger. (default: None)
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        dataloader: DataLoader,
        token_probabilities: torch.FloatTensor,
        pred_proportion: float = 0.15,
        mask_probability: float = 0.8,
        keep_probability: float = 0.1,
        rand_probability: float = 0.1,
        loss_fn: Optional[Loss] = None,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        num_epochs: Optional[int] = 1,
        num_gradient_accumulation_steps: Optional[int] = 1,
        max_gradient_norm: Optional[float] = None,
        use_cuda: Optional[bool] = False,
        local_rank: Optional[int] = 0,
        use_distributed: Optional[bool] = False,
        is_master: Optional[bool] = True,
        use_tqdm: Optional[bool] = True,
        logger: Optional[Logger] = None,
    ) -> None:
        super(SanhDistiller, self).__init__(
            student,
            teacher,
            dataloader,
            loss_fn,
            optimizer,
            lr_scheduler,
            num_epochs,
            num_gradient_accumulation_steps,
            max_gradient_norm,
            use_cuda,
            local_rank,
            use_distributed,
            is_master,
            use_tqdm,
            logger,
        )
        self.token_probabilities = token_probabilities
        self.pred_proportion = pred_proportion
        self.pred_probabilities = torch.FloatTensor(
            [mask_probability, keep_probability, rand_probability])

        # send token_probabilties and pred_probabilties to GPU
        if self.use_cuda:
            self.token_probabilities = self.token_probabilities.to(
                f'cuda:{self.local_rank}')
            self.pred_probabilities = self.pred_probabilities.to(
                f'cuda:{self.local_rank}')

        if self.loss_fn is None:
            self.loss_fn = SanhLoss()

        if self.optimizer is None:
            self.optimizer = optim.Adam(
                self.student.parameters(),
                lr=5e-4,
                eps=1e-6,
                betas=(0.9, 0.98)
            )

        if self.lr_scheduler is None:
            num_steps_epoch = len(self.dataloader)
            num_train_steps = math.ceil(
                num_steps_epoch / self.num_gradient_accumulation_steps * self.num_epochs)
            num_warmup_steps = math.ceil(num_train_steps * 0.05)

            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(0.0, float(num_train_steps - current_step) / float(max(1, num_train_steps - num_warmup_steps)))

            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lr_lambda,
                last_epoch=-1
            )

    def distill(self):
        # put student in train mode and teacher in eval mode
        self.student.train()
        self.teacher.eval()

        # keep track of the last loss
        self.last_loss = 0

        for epoch in range(self.num_epochs):
            # synchronize all processes
            if self.use_distributed:
                torch.distributed.barrier()

            if self.is_master and self.logger is not None:
                self.logger.info(
                    f'Starting with epoch {epoch+1}/{self.num_epochs}')

            # initialize the progress bar
            if self.use_tqdm:
                pbar = tqdm(
                    desc=f'Distilling [epoch {epoch+1}/{self.num_epochs}]',
                    total=len(self.dataloader),
                    unit='batch',
                    leave=False,
                )

            for step, batch in enumerate(self.dataloader):
                # unpack batch
                sequences, lengths = batch

                # send input and target to GPU
                if self.use_cuda:
                    sequences = sequences.to(f'cuda:{self.local_rank}')
                    lengths = lengths.to(f'cuda:{self.local_rank}')

                # prepare the batch
                sequences, attention_mask, mlm_target = self.prepare_batch(
                    sequences, lengths)

                # forward pass
                student_output = self.student(sequences, attention_mask=attention_mask)
                student_logits = student_output.logits
                student_hidden_states = student_output.hidden_states
                with torch.no_grad():
                    teacher_output = self.teacher(sequences, attention_mask=attention_mask)
                    teacher_logits = teacher_output.logits
                    teacher_hidden_states = teacher_output.hidden_states

                # select and reshape the logits
                logits_mask = attention_mask.unsqueeze(
                    -1).expand_as(student_logits)
                student_logits_masked = torch.masked_select(
                    student_logits, logits_mask)
                teacher_logits_masked = torch.masked_select(
                    teacher_logits, logits_mask)
                student_logits_masked = student_logits_masked.view(
                    -1, student_logits.size(-1))
                teacher_logits_masked = teacher_logits_masked.view(
                    -1, student_logits.size(-1))

                # select and reshape the hidden states
                student_hidden_states = student_hidden_states[-1]
                teacher_hidden_states = teacher_hidden_states[-1]
                hidden_states_mask = attention_mask.unsqueeze(
                    -1).expand_as(student_hidden_states)
                student_hidden_states_masked = torch.masked_select(
                    student_hidden_states, hidden_states_mask)
                teacher_hidden_states_masked = torch.masked_select(
                    teacher_hidden_states, hidden_states_mask)
                student_hidden_states_masked = student_hidden_states_masked.view(
                    -1, student_hidden_states.size(-1))
                teacher_hidden_states_masked = teacher_hidden_states_masked.view(
                    -1, student_hidden_states.size(-1))

                # compute the cosine embedding target
                cosine_emb_target = teacher_hidden_states_masked.new_ones(
                    teacher_hidden_states_masked.size(0))

                # compute the loss
                loss = self.loss_fn(
                    input1=student_logits_masked,
                    input2=teacher_logits_masked,
                    input3=student_logits.view(-1, student_logits.size(-1)),
                    input4=student_hidden_states_masked,
                    input5=teacher_hidden_states_masked,
                    target1=mlm_target.view(-1),
                    target2=cosine_emb_target
                )
                self.last_loss = loss.item()

                if self.use_distributed:
                    loss = loss.mean()

                # rescale the loss
                loss /= self.num_gradient_accumulation_steps

                # backward pass
                loss.backward()

                if step % self.num_gradient_accumulation_steps == 0:
                    # clip the gradient
                    if self.max_gradient_norm is not None:
                        clip_grad_norm_(
                            self.student.parameters(),
                            self.max_gradient_norm
                        )

                    # update the parameters
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    # clear all gradients
                    self.optimizer.zero_grad()

                # update the progress bar
                if self.use_tqdm:
                    pbar.update()
                    pbar.set_postfix({'last_loss': self.last_loss})

            # close the progress bar
            if self.use_tqdm:
                pbar.close()

    def prepare_batch(
        self,
        sequences: torch.LongTensor,
        lengths: torch.LongTensor
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        # compute the attention mask
        batch_size, max_seq_len = sequences.size()
        attention_mask = (torch.arange(
            max_seq_len, dtype=torch.long, device=lengths.device) < lengths[:, None])

        # prepare the target
        target = sequences.clone().detach()

        # get the token probabilities of the sequences in the batch
        _token_probabilities = self.token_probabilities[sequences.view(-1)]

        # compute the number of targets (tokens for which a prediction needs to be made)
        num_targets = math.ceil(self.pred_proportion * lengths.sum().item())

        # compute the prediction mask
        target_idxs = torch.multinomial(
            _token_probabilities / _token_probabilities.sum(), num_targets, replacement=False)
        pred_mask = torch.zeros(batch_size * max_seq_len,
                                dtype=torch.bool, device=sequences.device)
        pred_mask[target_idxs] = 1
        pred_mask = pred_mask.view(batch_size, max_seq_len)
        pred_mask[sequences ==
                  self.dataloader.dataset.special_tokens_map['pad_token']] = 0

        # compute the prediction tokens
        sequences_keep = sequences[pred_mask]
        sequences_rand = sequences_keep.clone().random_(
            self.dataloader.dataset._tokenizer.get_vocab_size())
        sequences_mask = sequences_keep.clone().fill_(
            self.dataloader.dataset.special_tokens_map['mask_token'])
        pred_idxs = torch.multinomial(
            self.pred_probabilities, len(sequences_keep), replacement=True)
        pred_tokens = sequences_mask * (pred_idxs == 0).long() + sequences_keep * (
            pred_idxs == 1).long() + sequences_rand * (pred_idxs == 2).long()

        # copy the prediction tokens into the sequences, given the prediction mask
        sequences = sequences.masked_scatter(pred_mask, pred_tokens)

        # ignore tokens that are not in the prediction mask
        target[~pred_mask] = -100

        return sequences, attention_mask, target
