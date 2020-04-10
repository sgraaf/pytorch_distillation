#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import math
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from tokenizers import BertWordPieceTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (BertForMaskedLM, DistilBertConfig,
                          DistilBertForMaskedLM)

from torch_distillation import (DistributedDistiller, GroupedBatchSampler,
                                KDLoss, LanguageModelingDataset, quantize)

# initialize the logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s',
                    datefmt='%Y%m%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# suppress transformers logging
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.configuration_utils').setLevel(logging.ERROR)

MIN_SEQUENCE_LEN = 12
MAX_SEQUENCE_LEN = 512
N_EPOCHS = 3
BATCH_SIZE = 5
LEARNING_RATE = 5e-4
EPSILON = 1e-6
WARMUP_PROP = 0.05
WEIGHT_DECAY = 0.0
TEMPERATURE = 2.0
N_GRAD_ACCUMULATION_STEPS = 50
MAX_GRAD_NORM = 5.0
SEED = 42

def main():
    parser = ArgumentParser('Distributed distillation example')
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank')
    params = parser.parse_args()
    params.is_master = params.local_rank == 0

    # initialize multi-GPU
    if params.is_master:
        logger.info('Initializing PyTorch distributed')
    torch.cuda.set_device(params.local_rank)                 
    torch.distributed.init_process_group(backend='nccl', init_method='env://')  

    # set seed(s)
    if params.is_master:
        logger.info('Setting random seed(s)')
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # root dir
    root = Path(__file__).resolve().parent

    # initialize the student
    if params.is_master:
        logger.info('Initializing the student')
    student_config_file = root / 'config.json'
    student_weights_file = root / 'weights.pth'
    student_config = DistilBertConfig.from_pretrained(student_config_file)
    student = DistilBertForMaskedLM.from_pretrained(
        student_weights_file, config=student_config)

    # initialize the teacher
    if params.is_master:
        logger.info('Initializing the teacher')
    teacher = BertForMaskedLM.from_pretrained('bert-base-uncased')

    # initialize the tokenizer
    if params.is_master:
        logger.info('Initializing the tokenizer')
    tokenizer = BertWordPieceTokenizer(str(root / 'vocab.txt'))

    # initialize the dataset
    if params.is_master:
        logger.info('Initializing the dataset')
    dataset = LanguageModelingDataset(
        path=root / 'small.txt',
        tokenizer=tokenizer,
        do_tokenize=True,
        min_sequence_len=MIN_SEQUENCE_LEN,
        max_sequence_len=MAX_SEQUENCE_LEN
    )

    # initialize the sampler
    if params.is_master:
        logger.info('Initializing the sampler')
    group_bins = list(range(3, dataset.max_sequence_len, 4))
    group_idxs = quantize(dataset.lengths, group_bins)
    base_sampler = DistributedSampler(dataset)
    sampler = GroupedBatchSampler(
        sampler=base_sampler,
        group_idxs=group_idxs,
        batch_size=BATCH_SIZE,
        drop_last=False
    )

    # initialize the dataloader
    if params.is_master:
        logger.info('Initializing the dataloader')
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=dataset.sequences_collate_fn
    )

    # initialize the loss function
    if params.is_master:
        logger.info('Initializing the loss function')
    loss_fn = KDLoss(temperature=TEMPERATURE, reduction='batchmean')
    def loss_fn_wrapper(input, target):
        return loss_fn(input[0], target[0])

    # initialize the optimizer
    if params.is_master:
        logger.info('Initializing the optimizer')
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': WEIGHT_DECAY
        }, {
            'params': [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=EPSILON
    )

    # initialize the scheduler
    if params.is_master:
        logger.info('Initializing the scheduler')
    n_steps_epoch = len(dataloader)
    n_train_steps = int(n_steps_epoch / N_GRAD_ACCUMULATION_STEPS * N_EPOCHS) + 1
    n_warmup_steps = math.ceil(n_train_steps * WARMUP_PROP)

    def lr_lambda(current_step):
        if current_step < n_warmup_steps:
            return float(current_step) / float(max(1, n_warmup_steps))
        return max(0.0, float(n_train_steps - current_step) / float(max(1, n_train_steps - n_warmup_steps)))
    scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lr_lambda,
        last_epoch=-1
    )                                     

    # initialize the distiller
    if params.is_master:
        logger.info('Initializing the distiller')
    distiller = DistributedDistiller(
        student=student,
        teacher=teacher,
        dataloader=dataloader,
        loss_fn=loss_fn_wrapper,
        optimizer=optimizer,
        num_epochs=N_EPOCHS,
        local_rank=params.local_rank,
        is_master=params.is_master,
        num_gradient_accumulation_steps=N_GRAD_ACCUMULATION_STEPS,
        max_gradient_norm=MAX_GRAD_NORM,
        scheduler=scheduler,
        use_tqdm=True,
        logger=logger,
    )

    # start the distillation
    if params.is_master:
        logger.info('Starting the distillation')
    distiller.distill()

    # save the student model weights
    if params.is_master:
        logger.info('Saving the student model weights')
        torch.save(student.state_dict(), './distilled_bert.pth')


if __name__ == '__main__':
    main()
