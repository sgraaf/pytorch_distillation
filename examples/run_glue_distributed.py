#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import logging
import math
import random
from argparse import ArgumentParser
from logging import Logger
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from tokenizers import BertWordPieceTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from torch_distillation import GLUE_TASKS, GLUE_TASKS_MAPPING, GLUETaskDataset

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

TRAINED_CONFIG_FILE_TEMPLATE = '{model_name}_{task}_config.json'
TRAINED_WEIGHTS_FILE_TEMPLATE = '{model_name}_{task}_weights.pth'
RESULTS_FILE_TEMPLATE = '{model_name}_{task}_results.json'


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    return (predictions == targets).mean()


def f1(predictions: np.ndarray, targets: np.ndarray) -> float:
    return f1_score(y_true=targets, y_pred=predictions)


def pearson(predictions: np.ndarray, targets: np.ndarray) -> float:
    return pearsonr(predictions, targets)[0]


def spearman(predictions: np.ndarray, targets: np.ndarray) -> float:
    return spearmanr(predictions, targets)[0]


def matthews(predictions: np.ndarray, targets: np.ndarray) -> float:
    return matthews_corrcoef(targets, predictions)


def train(
    task: str,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    num_epochs: int,
    lr_scheduler: Optional[LRScheduler] = None,
    num_gradient_accumulation_steps: Optional[int] = 1,
    max_gradient_norm: Optional[float] = None,
    use_cuda: Optional[bool] = False,
    local_rank: Optional[int] = 0,
    use_distributed: Optional[bool] = False,
    is_master: Optional[bool] = True,
    use_tqdm: Optional[bool] = True,
    logger: Optional[Logger] = None,
) -> None:
    # put model in train mode
    model.train()

    # keep track of the last loss
    last_loss = 0

    for epoch in range(num_epochs):
        # synchronize all processes
        if use_distributed:
            torch.distributed.barrier()

        if is_master and logger is not None:
            logger.info(f'{task} - Starting with epoch {epoch+1}/{num_epochs}')

        # initialize the progress bar
        if is_master and use_tqdm:
            pbar = tqdm(
                desc=f'Training {task} [epoch {epoch+1}/{num_epochs}]',
                total=len(dataloader),
                unit='batch',
            )

        for step, batch in enumerate(dataloader):
            # unpack batch
            sequences, attention_masks, labels = batch

            # send sequences, attention_masks and labels to GPU
            if use_cuda:
                sequences = sequences.to(f'cuda:{local_rank}')
                attention_masks = attention_masks.to(f'cuda:{local_rank}')
                labels = labels.to(f'cuda:{local_rank}')

            # forward pass (loss computation included)
            outputs = model(
                input_ids=sequences,
                attention_mask=attention_masks,
                labels=labels
            )
            loss = outputs[0]
            last_loss = loss.item()

            if use_distributed:
                loss = loss.mean()

            # rescale the loss
            loss /= num_gradient_accumulation_steps

            # backward pass
            loss.backward()

            if step % num_gradient_accumulation_steps == 0:
                # clip the gradient
                if max_gradient_norm is not None:
                    clip_grad_norm_(
                        model.parameters(),
                        max_gradient_norm
                    )

                # update the parameters
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                # clear all gradients
                optimizer.zero_grad()

            # update the progress bar
            if is_master and use_tqdm:
                pbar.update()
                pbar.set_postfix({'last_loss': last_loss})

        # close the progress bar
        if is_master and use_tqdm:
            pbar.close()


def evaluate(
    task: str,
    model: nn.Module,
    dataloader: DataLoader,
    use_cuda: Optional[bool] = False,
    local_rank: Optional[int] = 0,
    use_tqdm: Optional[bool] = True,
) -> None:
    # put model in eval mode
    model.eval()

    # initialize metrics
    num_eval_steps = 0
    eval_loss = 0
    predictions = np.empty((0, model.num_labels))
    targets = np.empty((0,))

    # initialize the progress bar
    if use_tqdm:
        pbar = tqdm(
            desc=f'Evaluating {task}',
            total=len(dataloader),
            unit='batch',
        )

    for step, batch in enumerate(dataloader):
        # unpack batch
        sequences, attention_masks, labels = batch

        # send sequences, attention_masks and labels to GPU
        if use_cuda:
            sequences = sequences.to(f'cuda:{local_rank}')
            attention_masks = attention_masks.to(f'cuda:{local_rank}')
            labels = labels.to(f'cuda:{local_rank}')

        with torch.no_grad():
            # forward pass (loss compution included)
            outputs = model(
                input_ids=sequences,
                attention_mask=attention_masks,
                labels=labels
            )
            loss, logits = outputs[:2]

        # update metrics
        eval_loss += loss.mean().item()
        num_eval_steps += 1
        predictions = np.append(
            arr=predictions,
            values=logits.detach().cpu().numpy(),
            axis=0
        )
        targets = np.append(
            arr=targets,
            values=labels.detach().cpu().numpy(),
            axis=0
        )

        # update the progress bar
        if use_tqdm:
            pbar.update()

    # close the progress bar
    if use_tqdm:
        pbar.close()

    # compute results
    eval_loss /= num_eval_steps
    if GLUE_TASKS_MAPPING[task]['type'] == 'classification':
        predictions = np.argmax(predictions, axis=1)
    elif GLUE_TASKS_MAPPING[task]['type'] == 'regression':
        predictions = np.squeeze(predictions)

    if task == 'CoLA':
        return {'mcc': matthews(predictions, targets)}
    elif task in {'MNLI', 'MNLI-MM', 'SST-2', 'QNLI', 'RTE', 'WNLI'}:
        return {'acc': accuracy(predictions, targets)}
    elif task in {'MRPC', 'QQP'}:
        return {'acc': accuracy(predictions, targets), 'f1': f1(predictions, targets)}
    elif task == 'STS-B':
        return {'pearsonr': pearson(predictions, targets), 'spearmanr': spearman(predictions, targets)}


def main():
    parser = ArgumentParser('GLUE evaluation example')
    parser.add_argument('--glue_dir', type=str, metavar='PATH',
                        required=True, help='Path to directory containing the GLUE tasks data.')
    parser.add_argument('--output_dir', type=str, metavar='PATH', required=True,
                        help='Path to the output directory (for logs, checkpoints, parameters, etc.).')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Overwrite output_dir if it already exists.')
    parser.add_argument('--task_name', type=str, default=None, choices=GLUE_TASKS,
                        help='The specific GLUE task to train and/or evaluate on.')
    parser.add_argument('--do_train', action='store_true',
                        help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true',
                        help='Whether to run eval (on the dev set).')
    parser.add_argument('--config_file', type=str, metavar='PATH',
                        required=True, help='Path to the model configuration.')
    parser.add_argument('--weights_file', type=str, metavar='PATH',
                        required=True, help='Path to the model initialization weights.')
    parser.add_argument('--tokenizer_vocab_file', type=str, metavar='PATH',
                        required=True, help='Path to the tokenizer vocabulary.')
    parser.add_argument('--max_sequence_len', type=int, default=128,
                        metavar='N', help='The maximum length of a sequence.')
    parser.add_argument('--do_lower_case', action='store_true',
                        help='Whether to lowercase the input when tokenizing.')
    parser.add_argument('-n', '--num_epochs', type=int, default=3,
                        metavar='N', help='The number of distillation epochs.')
    parser.add_argument('--train_batch_size', type=int,
                        default=8, metavar='N', help='The batch size used during training.')
    parser.add_argument('--eval_batch_size', type=int,
                        default=8, metavar='N', help='The batch size used during evaluation.')
    parser.add_argument('-lr', '--learning_rate', type=float,
                        default=2e-5, metavar='F', help='The initial learning rate.')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        metavar='F', help="Adam's epsilon.")
    parser.add_argument('--warmup_prop', type=float, default=0.05,
                        metavar='F', help='Linear warmup proportion.')
    parser.add_argument('--num_gradient_accumulation_steps', type=int, default=1, metavar='N',
                        help='The number of gradient accumulation steps (for larger batch sizes).')
    parser.add_argument('--max_gradient_norm', type=float,
                        default=1.0, metavar='F', help='The maximum gradient norm.')
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
    params.glue_dir = Path(params.glue_dir)
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

    # initialize the tokenizer
    if params.is_master:
        logger.info('Initializing the tokenizer')
    tokenizer = BertWordPieceTokenizer(
        params.tokenizer_vocab_file,
        lowercase=params.do_lower_case
    )

    # enable truncation and padding
    tokenizer.enable_truncation(params.max_sequence_len)
    tokenizer.enable_padding(max_length=params.max_sequence_len)

    # go over each task
    tasks = [params.task_name] if params.task_name is not None else GLUE_TASKS
    for task in tasks:
        # prepare the GLUE task
        if params.is_master:
            logger.info(f'Preparing the {task} GLUE task')
        task_output_dir = params.output_dir / task / params.seed
        
        # make task_output_dir
        if task_output_dir.is_dir() and not params.force:
            raise ValueError(f'Task output directory {task_output_dir} already exists. Use `--force` if you want to overwrite it.')
        if params.is_master:
            task_output_dir.mkdir(parents=True, exist_ok=params.force)

        # initialize the model
        if params.is_master:
            logger.info(f'{task} - Initializing the model')
        config = DistilBertConfig.from_pretrained(
            params.config_file,
            num_labels=len(GLUE_TASKS_MAPPING[task]['labels']),
            finetuning_task=task
        )
        model = DistilBertForSequenceClassification.from_pretrained(
            params.weights_file,
            config=config
        )

        # send model to GPU
        if params.use_cuda:
            model = model.to(f'cuda:{params.local_rank}')

        # perform the training
        if params.do_train:
            # initialize the training dataset
            if params.is_master:
                logger.info(f'{task} - Initializing the training dataset')
            train_dataset = GLUETaskDataset(
                task=task,
                glue_dir=params.glue_dir,
                split='train',
                tokenizer=tokenizer
            )

            # initialize the sampler
            if params.is_master:
                logger.info(f'{task} - Initializing the training sampler')
            train_sampler = DistributedSampler(train_dataset) if params.use_distributed else RandomSampler(train_dataset)

            # initialize the dataloader
            if params.is_master:
                logger.info(f'{task} - Initializing the training dataloader')
            train_dataloader = DataLoader(
                dataset=train_dataset,
                sampler=train_sampler,
                batch_size=params.train_batch_size
            )

            # initialize the optimizer
            if params.is_master:
                logger.info(f'{task} - Initializing the optimizer')
            optimizer = optim.Adam(
                model.parameters(),
                lr=params.learning_rate,
                eps=params.epsilon,
                betas=(0.9, 0.98)
            )

            # initialize the learning rate scheduler
            if params.is_master:
                logger.info(f'{task} - Initializing the learning rate scheduler')
            num_steps_epoch = len(train_dataloader)
            num_train_steps = math.ceil(num_steps_epoch / params.num_gradient_accumulation_steps * params.num_epochs)
            num_warmup_steps = math.ceil(num_train_steps * params.warmup_prop)

            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(0.0, float(num_train_steps - current_step) / float(max(1, num_train_steps - num_warmup_steps)))

            lr_scheduler = optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=lr_lambda,
                last_epoch=-1
            )

            # initialize distributed data parallel (DDP)
            if params.use_distributed:
                model = DDP(
                    model,
                    device_ids=[params.local_rank],
                    output_device=params.local_rank
                )

            # start training
            if params.is_master:
                logger.info(f'{task} - Starting the training')
            train(
                task=task,
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                num_epochs=params.num_epochs,
                lr_scheduler=lr_scheduler,
                num_gradient_accumulation_steps=params.num_gradient_accumulation_steps,
                max_gradient_norm=params.max_gradient_norm,
                use_cuda=params.use_cuda,
                local_rank=params.local_rank,
                use_distributed=params.use_distributed,
                is_master=params.is_master,
                use_tqdm=True,
                logger=logger
            )

            # save the finetuned model
            if params.is_master:
                # take care of distributed training
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.config.architectures = [model_to_save.__class__.__name__]

                logger.info(f'{task} - Saving the finetuned model config')
                json.dump(
                    vars(model_to_save.config),
                    open(
                        task_output_dir / TRAINED_CONFIG_FILE_TEMPLATE.format(
                            model_name=model_to_save.__class__.__name__,
                            task=task
                        ), mode='w'
                    ), indent=4,
                    sort_keys=True
                )

                logger.info(f'{task} - Saving the finetuned model weights')
                torch.save(
                    model_to_save.state_dict(),
                    task_output_dir / TRAINED_WEIGHTS_FILE_TEMPLATE.format(
                        model_name=model_to_save.__class__.__name__,
                        task=task
                    )
                )

                # reload the model
                if params.do_eval:
                    if params.is_master:
                        logger.info(f'{task} - Reloading the model')
                    config = DistilBertConfig.from_pretrained(
                        str(task_output_dir / f'{model_to_save.__class__.__name__}_{task}_config.json'),
                        num_labels=len(GLUE_TASKS_MAPPING[task]['labels']),
                        finetuning_task=task
                    )
                    model = DistilBertForSequenceClassification.from_pretrained(
                        str(task_output_dir / f'{model_to_save.__class__.__name__}_{task}_weights.pth'),
                        config=config
                    )

                    if params.use_cuda:
                        model = model.to(f'cuda:{params.local_rank}')

        # perform the evaluation
        if params.do_eval and params.is_master:
            logger.info(f'{task} - Initializing the evaluation dataset')
            eval_datasets = [
                GLUETaskDataset(
                    task=task,
                    glue_dir=params.glue_dir,
                    split='dev',
                    tokenizer=tokenizer
                )
            ]

            # hot fix for MNLI-MM
            if task == 'MNLI':
                eval_datasets.append(
                    GLUETaskDataset(
                        task='MNLI-MM',
                        glue_dir=params.glue_dir,
                        split='dev',
                        tokenizer=tokenizer
                    )
                )

            for eval_dataset in eval_datasets:
                # initialize the sampler
                logger.info(f'{eval_dataset.task} - Initializing the evaluation sampler')
                eval_sampler = SequentialSampler(eval_dataset)

                # initialize the dataloader
                logger.info(f'{eval_dataset.task} - Initializing the evaluation dataloader')
                eval_dataloader = DataLoader(
                    dataset=eval_dataset,
                    sampler=eval_sampler,
                    batch_size=params.eval_batch_size
                )

                # start training
                if params.is_master:
                    logger.info(f'{eval_dataset.task} - Starting the evaluation')
                results = evaluate(
                    task=task,
                    model=model,
                    dataloader=eval_dataloader,
                    use_cuda=params.use_cuda,
                    local_rank=params.local_rank,
                    use_tqdm=True,
                )

                # log results
                logger.info(f'{eval_dataset.task} - Evaluation results:')
                for key in results:
                    logger.info(f'{eval_dataset.task} -  {key}: {results[key]}')

                # dump results
                json.dump(
                    results,
                    open(task_output_dir / RESULTS_FILE_TEMPLATE.format(model_name=model.__class__.__name__, task=task), 'w'),
                    indent=4
                )

        if params.is_master:
            logger.info(f'Done with the {task} GLUE task')


if __name__ == '__main__':
    main()
