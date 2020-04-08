#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import math
from pathlib import Path

import torch
from tokenizers import BertWordPieceTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler
from transformers import (BertForMaskedLM, DistilBertConfig,
                          DistilBertForMaskedLM)

from torch_distillation import (Distiller, GroupedBatchSampler, KDLoss,
                                LanguageModelingDataset, quantize)

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


def main():
    # root dir
    root = Path(__file__).resolve().parent

    # initialize the student
    logger.info('Initializing the student')
    student_config_file = root / 'config.json'
    student_weights_file = root / 'weights.pth'
    student_config = DistilBertConfig.from_pretrained(student_config_file)
    student = DistilBertForMaskedLM.from_pretrained(
        student_weights_file, config=student_config)

    # initialize the teacher
    logger.info('Initializing the teacher')
    teacher = BertForMaskedLM.from_pretrained('bert-base-uncased')

    # initialize the tokenizer
    logger.info('Initializing the tokenizer')
    tokenizer = BertWordPieceTokenizer(str(root / 'vocab.txt'))

    # initialize the dataset
    logger.info('Initializing the dataset')
    dataset = LanguageModelingDataset(
        path=root / 'small.txt',
        tokenizer=tokenizer,
        do_tokenize=True,
        min_sequence_len=MIN_SEQUENCE_LEN,
        max_sequence_len=MAX_SEQUENCE_LEN
    )

    # initialize the sampler
    logger.info('Initializing the sampler')
    group_bins = list(range(3, dataset.max_sequence_len, 4))
    group_idxs = quantize(dataset.lengths, group_bins)
    base_sampler = RandomSampler(dataset)
    sampler = GroupedBatchSampler(
        sampler=base_sampler,
        group_idxs=group_idxs,
        batch_size=BATCH_SIZE,
        drop_last=False
    )

    # initialize the dataloader
    logger.info('Initializing the dataloader')
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=dataset.sequences_collate_fn
    )

    # initialize the loss function
    logger.info('Initializing the loss function')
    loss_fn = KDLoss(temperature=TEMPERATURE, reduction='batchmean')
    def loss_fn_wrapper(input, target):
        return loss_fn(input[0], target[0])

    # initialize the optimizer
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
    logger.info('Initializing the scheduler')
    n_steps_epoch = len(dataloader)
    n_train_steps = n_steps_epoch * N_EPOCHS
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
    logger.info('Initializing the distiller')
    distiller = Distiller(
        student=student,
        teacher=teacher,
        dataloader=dataloader,
        loss_fn=loss_fn_wrapper,
        optimizer=optimizer,
        num_epochs=N_EPOCHS,
        scheduler=scheduler,
        use_cuda=True,
        logger=logger
    )

    # start the distillation
    logger.info('Starting the distillation')
    distiller.distill()

    # save the student model weights
    logger.info('Saving the student model weights')
    torch.save(student.state_dict(), './distilled_bert.pth')


if __name__ == '__main__':
    main()
