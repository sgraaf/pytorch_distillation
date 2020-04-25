#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import logging
import random
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers import (BertForMaskedLM, DistilBertConfig,
                          DistilBertForMaskedLM)

from torch_distillation import (GroupedBatchSampler, LanguageModelingDataset,
                                SanhDistiller, SanhLoss)
from torch_distillation.data import quantize

# initialize the logger
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# suppress transformers logging
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.configuration_utils').setLevel(logging.ERROR)


def main():
    parser = ArgumentParser('Distributed distillation example')
    parser.add_argument('--data_file', type=str, metavar='PATH',
                        required=True, help='Path to file containing the data (sequences).')
    parser.add_argument('--output_dir', type=str, metavar='PATH', required=True,
                        help='Path to the output directory (for logs, checkpoints, parameters, etc.).')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Overwrite output_dir if it already exists.')
    parser.add_argument('--student_config_file', type=str, metavar='PATH',
                        required=True, help='Path to the student model configuration.')
    parser.add_argument('--student_weights_file', type=str, default=None,
                        metavar='PATH', help='Path to the student model initialization weights.')
    parser.add_argument('--teacher_type', type=str, default=None,
                        choices={'bert-base-uncased'}, help='The pre-trained teacher model type to initialize.')
    parser.add_argument('--tokenizer_vocab_file', type=str, metavar='PATH',
                        required=True, help='Path to the tokenizer vocabulary.')
    parser.add_argument('--min_sequence_len', type=int, default=12,
                        metavar='N', help='The minimum length of a sequence.')
    parser.add_argument('--max_sequence_len', type=int, default=512,
                        metavar='N', help='The maximum length of a sequence.')
    parser.add_argument('--do_tokenize', action='store_true',
                        help='Whether to tokenize the input.')
    parser.add_argument('--do_lower_case', action='store_true',
                        help='Whether to lowercase the input when tokenizing.')
    parser.add_argument('-n', '--num_epochs', type=int, default=3,
                        metavar='N', help='The number of distillation epochs.')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=5, metavar='N', help='The batch size.')
    parser.add_argument('--lr', '--learning_rate', type=float,
                        default=5e-4, metavar='F', help='The initial learning rate.')
    parser.add_argument('--epsilon', type=float, default=1e-6,
                        metavar='F', help="Adam's epsilon.")
    parser.add_argument('--warmup_prop', type=float, default=0.05,
                        metavar='F', help='Linear warmup proportion.')
    parser.add_argument('--num_gradient_accumulation_steps', type=int, default=50, metavar='N',
                        help='The number of gradient accumulation steps (for larger batch sizes).')
    parser.add_argument('--max_gradient_norm', type=float,
                        default=5.0, metavar='F', help='The maximum gradient norm.')
    parser.add_argument('--soft_target_alpha', type=float, default=0.33,
                        metavar='F', help='The relative weight of the soft target loss.')
    parser.add_argument('--hard_target_alpha', type=float, default=0.33,
                        metavar='F', help='The relative weight of the hard target loss.')
    parser.add_argument('--cosine_emb_alpha', type=float, default=0.33,
                        metavar='F', help='The relative weight of the cosine embedding loss.')
    parser.add_argument('--seed', type=int, default=42,
                        metavar='N', help='Random seed.')
    parser.add_argument('-c', '--use_cuda', action='store_true',
                        help='Whether to use cuda or not.')
    parser.add_argument('-d', '--use_distributed', action='store_true',
                        help='Whether to use distributed training (distillation) or not.')
    parser.add_argument('--local_rank', type=int, default=-1,
                        metavar='N', help='Local process rank.')
    params = parser.parse_args()

    if not params.use_distributed:
        params.local_rank = 0
    params.is_master = params.local_rank == 0

    # make output_dir
    if Path(params.output_dir).is_dir() and not params.force:
        raise ValueError(
            f'Output directory {params.output_dir} already exists. Use `--force` if you want to overwrite it.')
    if params.is_master:
        Path(params.output_dir).mkdir(parents=True, exist_ok=params.force)

        # dump params
        json.dump(
            vars(params),
            open(Path(params.output_dir) / 'params.json', 'w'),
            indent=4,
            sort_keys=True
        )
    params.output_dir = Path(params.output_dir)

    # initialize multi-GPU
    if params.use_distributed:
        if params.is_master:
            logger.info('Initializing PyTorch distributed')
        torch.cuda.set_device(params.local_rank)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://'
        )

    # set seed(s)
    if params.is_master:
        logger.info('Setting random seed(s)')
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    if params.use_distributed:
        torch.cuda.manual_seed_all(params.seed)

    # initialize the student
    if params.is_master:
        logger.info('Initializing the student')
    student_config = DistilBertConfig.from_pretrained(
        params.student_config_file)
    student_config.output_hidden_states = True
    if params.student_weights_file is not None:
        student = DistilBertForMaskedLM.from_pretrained(
            params.student_weights_file,
            config=student_config
        )
    else:
        student = DistilBertForMaskedLM(student_config)

    # initialize the teacher
    if params.is_master:
        logger.info('Initializing the teacher')
    teacher = BertForMaskedLM.from_pretrained(
        params.teacher_type, output_hidden_states=True)

    # initialize the tokenizer
    if params.is_master:
        logger.info('Initializing the tokenizer')
    tokenizer = BertWordPieceTokenizer(
        params.tokenizer_vocab_file,
        lowercase=params.do_lower_case
    )

    # initialize the dataset
    if params.is_master:
        logger.info('Initializing the dataset')
    dataset = LanguageModelingDataset(
        path=params.data_file,
        tokenizer=tokenizer,
        do_tokenize=params.do_tokenize,
        min_sequence_len=params.min_sequence_len,
        max_sequence_len=params.max_sequence_len
    )

    # initialize the sampler
    if params.is_master:
        logger.info('Initializing the sampler')
    group_bins = list(range(3, params.max_sequence_len, 4))
    group_idxs = quantize(dataset.lengths, group_bins)
    sampler = GroupedBatchSampler(
        sampler=DistributedSampler(dataset) if params.use_distributed else RandomSampler(dataset),
        group_idxs=group_idxs,
        batch_size=params.batch_size,
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
    loss_fn = SanhLoss(
        alphas=(
            params.soft_target_alpha,
            params.hard_target_alpha,
            params.cosine_emb_alpha
        ),
        reduction=('batchmean', 'mean', 'mean')
    )

    # compute token counts
    if params.is_master:
        logger.info('Computing token counts')
    counter = Counter()
    for sequence in dataset.sequences:
        counter.update(sequence)
    token_counts = [0] * dataset._tokenizer.get_vocab_size()
    for k, v in counter.items():
        token_counts[k] = v
    del counter

    # compute token probabilities
    if params.is_master:
        logger.info('Computing token probabilities')
    token_probabilities = np.maximum(token_counts, 1) ** -0.7

    # give special tokens a zero probability
    for idx in dataset.special_tokens_map.values():
        token_probabilities[idx] = 0.0

    # convert to torch.FloatTensor
    token_probabilities = torch.FloatTensor(token_probabilities)

    # initialize the distiller
    if params.is_master:
        logger.info('Initializing the distiller')
    distiller = SanhDistiller(
        student=student,
        teacher=teacher,
        dataloader=dataloader,
        token_probabilities=token_probabilities,
        loss_fn=loss_fn,
        num_epochs=params.num_epochs,
        num_gradient_accumulation_steps=params.num_gradient_accumulation_steps,
        max_gradient_norm=params.max_gradient_norm,
        use_cuda=params.use_cuda,
        local_rank=params.local_rank,
        use_distributed=params.use_distributed,
        is_master=params.is_master,
        use_tqdm=True,
        logger=logger,
    )

    # start the distillation
    if params.is_master:
        logger.info('Starting the distillation')
    distiller.distill()

    # save the student model config and weights
    if params.is_master:
        logger.info('Saving the student model config')
        json.dump(
            vars(student.config),
            open(params.output_dir / 'distilled_bert_config.json', 'w'),
            indent=4,
            sort_keys=True
        )

        logger.info('Saving the student model weights')
        model_to_save = student.module if hasattr(student, 'module') else student  # Take care of distributed/parallel training
        torch.save(
            model_to_save.state_dict(),
            params.output_dir / 'distilled_bert_weights.pth'
        )


if __name__ == '__main__':
    main()
