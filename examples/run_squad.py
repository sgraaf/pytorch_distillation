#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import logging
import math
import os
import random
from argparse import ArgumentParser
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (BertTokenizer, DistilBertConfig,
                          DistilBertForQuestionAnswering,
                          squad_convert_examples_to_features)
from transformers.data import (SquadExample, SquadFeatures, SquadV1Processor,
                               squad_convert_examples_to_features)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits, squad_evaluate)
from transformers.data.processors.squad import SquadResult

# initialize the logger
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# suppress transformers logging
logging.getLogger('transformers.modeling_utils').setLevel(logging.WARNING)
logging.getLogger('transformers.configuration_utils').setLevel(logging.WARNING)
logging.getLogger('transformers.tokenization_utils').setLevel(logging.WARNING)
logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.WARNING)

TRAINED_CONFIG_FILE_TEMPLATE = '{model_name}_config.json'
TRAINED_WEIGHTS_FILE_TEMPLATE = '{model_name}_weights.pth'
RESULTS_FILE_TEMPLATE = '{model_name}_results.json'


def train(
    model: nn.Module,
    num_epochs: int,
    dataloader: DataLoader,
    optimizer: Optimizer,
    lr_scheduler: Optional[LRScheduler] = None,
    num_gradient_accumulation_steps: Optional[int] = 1,
    max_gradient_norm: Optional[float] = None,
    device: Optional[torch.device] = torch.device('cpu'),
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
            dist.barrier()

        if is_master and logger is not None:
            logger.info(f'Starting with epoch {epoch+1}/{num_epochs}')

        # initialize the progress bar
        if is_master and use_tqdm:
            pbar = tqdm(
                desc=f'Training [epoch {epoch+1}/{num_epochs}]',
                total=len(dataloader),
                unit='batch',
            )

        for step, batch in enumerate(dataloader):
            # unpack batch
            sequences, attention_masks, _, start_positions, end_positions, _, _, _ = batch

            # send sequences, attention_masks, start_positions and end_positions to device
            sequences = sequences.to(device)
            attention_masks = attention_masks.to(device)
            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)

            # forward pass (loss computation included)
            outputs = model(
                input_ids=sequences,
                attention_mask=attention_masks,
                start_positions=start_positions,
                end_positions=end_positions
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


def to_list(tensor: torch.Tensor) -> List[Any]:
    return tensor.detach().cpu().tolist()


def evaluate(
    output_dir: Path,
    model: nn.Module,
    tokenizer: BertTokenizer,
    dataloader: DataLoader,
    examples: List[SquadExample],
    features: List[SquadFeatures],
    max_answer_len: Optional[int] = 30,
    do_lower_case: Optional[bool] = False,
    device: Optional[torch.device] = torch.device('cpu'),
    local_rank: Optional[int] = 0,
    use_tqdm: Optional[bool] = True,
) -> Dict[str, float]:
    # put model in eval mode
    model.eval()

    # initialize metrics
    num_eval_steps = 0
    all_results = []

    # initialize the progress bar
    if use_tqdm:
        pbar = tqdm(
            desc=f'Evaluating',
            total=len(dataloader),
            unit='batch',
        )

    for step, batch in enumerate(dataloader):
        # unpack batch
        sequences, attention_masks, _, feature_indices, _, _ = batch

        # send sequences, attention_masks and feature_indices to device
        sequences = sequences.to(device)
        attention_masks = attention_masks.to(device)
        feature_indices = feature_indices.to(device)

        with torch.no_grad():
            # forward pass (loss computation included)
            outputs = model(
                input_ids=sequences,
                attention_mask=attention_masks,
                output_hidden_states=False
            )

        # compute results for batch
        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]
            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)
            all_results.append(result)

        # update the progress bar
        if use_tqdm:
            pbar.update()

    # close the progress bar
    if use_tqdm:
        pbar.close()

    # compute predictions
    predictions = compute_predictions_logits(
        all_examples=examples,
        all_features=features,
        all_results=all_results,
        n_best_size=20,
        max_answer_length=max_answer_len,
        do_lower_case=do_lower_case,
        output_prediction_file=str(output_dir / 'predictions.json'),
        output_nbest_file=str(output_dir / 'n_best_predictions.json'),
        output_null_log_odds_file=None,
        verbose_logging=False,
        version_2_with_negative=False,
        null_score_diff_threshold=0.0,
        tokenizer=tokenizer
    )

    # compute the Exact Match (EM) and F1-scores.
    results = squad_evaluate(examples, predictions)
    
    return results


def load_and_cache_examples(
    squad_dir: Path,
    split: str,
    tokenizer: BertTokenizer,
    max_sequence_len: Optional[int] = 384,
    max_query_len: Optional[int] = 64,
    doc_stride: Optional[int] = 128,
    output_examples: Optional[bool] = False,
    overwrite_cache: Optional[bool] = False,
    use_distributed: Optional[bool] = False,
    is_master: Optional[bool] = True
) -> Union[TensorDataset, Tuple[TensorDataset, List[SquadExample], List[SquadFeatures]]]:
    if use_distributed and not is_master and split == 'train':
        dist.barrier()

    # get the cached tensor paths
    cached_dataset_path = squad_dir / \
        f'cached_{split}_{tokenizer.__class__.__name__}_{str(max_sequence_len)}_dataset.pth'
    cached_examples_path = squad_dir / \
        f'cached_{split}_{tokenizer.__class__.__name__}_{str(max_sequence_len)}_examples.pth'
    cached_features_path = squad_dir / \
        f'cached_{split}_{tokenizer.__class__.__name__}_{str(max_sequence_len)}_features.pth'

    if (
        cached_dataset_path.exists()
        and cached_examples_path.exists()
        and cached_features_path.exists()
        and not overwrite_cache
    ):
        # load tensors from cache
        dataset = torch.load(cached_dataset_path)
        examples = torch.load(cached_examples_path)
        features = torch.load(cached_features_path)
    else:
        # load and process the data
        processor = SquadV1Processor()
        if split == 'train':
            examples = processor.get_train_examples(str(squad_dir))
        else:
            examples = processor.get_dev_examples(str(squad_dir))

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_sequence_len,
            doc_stride=doc_stride,
            max_query_length=max_query_len,
            is_training=split == 'train',
            return_dataset='pt',
            threads=os.cpu_count(),
        )

        # save tensors to cache
        if is_master:
            torch.save(dataset, cached_dataset_path)
            torch.save(examples, cached_examples_path)
            torch.save(features, cached_features_path)

    if use_distributed and is_master and split == 'train':
        dist.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = ArgumentParser('SQuAD evaluation example')
    parser.add_argument('--squad_dir', type=str, metavar='PATH', required=True,
                        help='Path to directory containing the SQuAD data (JSON-files).')
    parser.add_argument('--output_dir', type=str, metavar='PATH', required=True,
                        help='Path to the output directory (for logs, checkpoints, parameters, etc.).')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Overwrite `output_dir` if it already exists.')
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
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='Overwrite the cache if it already exists.')
    parser.add_argument('--max_sequence_len', type=int, default=384,
                        metavar='N', help='The maximum length of a sequence.')
    parser.add_argument('--max_query_len', type=int, default=64,
                        help='The maximum length of a question.')
    parser.add_argument('--max_answer_len', type=int,
                        default=30, help='The maximum length of an answer.')
    parser.add_argument('--doc_stride', type=int, default=128,
                        help='The stride to take between chunks when splitting a large document.')
    parser.add_argument('--do_lower_case', action='store_true',
                        help='Whether to lowercase the input when tokenizing.')
    parser.add_argument('-n', '--num_epochs', type=int, default=3,
                        metavar='N', help='The number of distillation epochs.')
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=8,
                        metavar='N', help='The batch size per GPU used during training.')
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=8,
                        metavar='N', help='The batch size per GPU used during evaluation.')
    parser.add_argument('--learning_rate', default=3e-5,
                        type=float, help='The initial learning rate for Adam.')
    parser.add_argument('--epsilon', default=1e-8,
                        type=float, help="Adam's epsilon.")
    parser.add_argument('--num_warmup_steps', default=0,
                        type=int, help='Linear warmup over `warmup_steps`.')
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

    if params.doc_stride >= params.max_sequence_len - params.max_query_len:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            'examples. This could result in errors when building features from the examples. Please reduce the doc '
            'stride or increase the maximum length to ensure the features are correctly built.'
        )

    if not params.use_distributed:
        params.local_rank = 0
        params.train_batch_size = params.per_gpu_train_batch_size
        params.eval_batch_size = params.per_gpu_eval_batch_size
    else:
        params.num_gpus = torch.cuda.device_count()
        params.train_batch_size = params.per_gpu_train_batch_size * params.num_gpus
        params.eval_batch_size = params.per_gpu_eval_batch_size * params.num_gpus
    params.is_master = params.local_rank == 0

    if params.use_cuda:
        device = torch.device('cuda', params.local_rank)
    else:
        device = torch.device('cpu')

    if Path(params.output_dir).is_dir() and not params.force:
        raise ValueError(f'Output directory {params.output_dir} already exists. Use `--force` if you want to overwrite it.')
    if params.is_master:
        Path(params.output_dir).mkdir(parents=True, exist_ok=params.force)

        # dump params
        json.dump(
            vars(params),
            open(Path(params.output_dir) / 'params.json', 'w'),
            indent=4,
            sort_keys=True
        )
    params.squad_dir = Path(params.squad_dir)
    params.output_dir = Path(params.output_dir)
    params.device = device

    # initialize multi-GPU
    if params.use_distributed:
        if params.is_master:
            logger.info('Initializing PyTorch distributed')
        torch.cuda.set_device(params.local_rank)
        dist.init_process_group(
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
    tokenizer = BertTokenizer.from_pretrained(
        params.tokenizer_vocab_file,
        do_lower_case=params.do_lower_case
    )

    # initialize the model
    if params.is_master:
        logger.info('Initializing the model')
    config = DistilBertConfig.from_pretrained(params.config_file)
    model = DistilBertForQuestionAnswering.from_pretrained(
        params.weights_file, config=config)

    # send model to device
    model = model.to(params.device)

    # perform the training
    if params.do_train:
        # initialize the training dataset
        if params.is_master:
            logger.info('Initializing the training dataset')
        train_dataset = load_and_cache_examples(
            squad_dir=params.squad_dir,
            split='train',
            tokenizer=tokenizer,
            max_sequence_len=params.max_sequence_len,
            max_query_len=params.max_query_len,
            doc_stride=params.doc_stride,
            output_examples=False,
            overwrite_cache=params.overwrite_cache,
            is_master=params.is_master
        )

        # initialize the sampler
        if params.is_master:
            logger.info('Initializing the training sampler')
        train_sampler = DistributedSampler(
            train_dataset) if params.use_distributed else RandomSampler(train_dataset)

        # initialize the dataloader
        if params.is_master:
            logger.info('Initializing the training dataloader')
        train_dataloader = DataLoader(
            dataset=train_dataset,
            sampler=train_sampler,
            batch_size=params.train_batch_size
        )

        # initialize the optimizer
        if params.is_master:
            logger.info('Initializing the optimizer')
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params.learning_rate,
            eps=params.epsilon,
        )

        # initialize the learning rate scheduler
        if params.is_master:
            logger.info('Initializing the learning rate scheduler')
        num_steps_epoch = len(train_dataloader)
        num_train_steps = math.ceil(
            num_steps_epoch / params.num_gradient_accumulation_steps * params.num_epochs)
        num_warmup_steps = params.num_warmup_steps

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
            if params.is_master:
                logger.info('Initializing DDP')
            model = DDP(
                model,
                device_ids=[params.local_rank],
                output_device=params.local_rank,
                find_unused_parameters=True
            )

        # start training
        if params.is_master:
            logger.info('Starting the training')
        train(
            model=model,
            num_epochs=params.num_epochs,
            dataloader=train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            num_gradient_accumulation_steps=params.num_gradient_accumulation_steps,
            max_gradient_norm=params.max_gradient_norm,
            device=params.device,
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
            model_to_save.config.architectures = [
                model_to_save.__class__.__name__]

            logger.info('Saving the finetuned model config')
            json.dump(
                vars(model_to_save.config),
                open(
                    params.output_dir /
                    TRAINED_CONFIG_FILE_TEMPLATE.format(
                        model_name=model_to_save.__class__.__name__),
                    mode='w'
                ), indent=4,
                sort_keys=True
            )

            logger.info('Saving the finetuned model weights')
            torch.save(
                model_to_save.state_dict(),
                params.output_dir /
                TRAINED_WEIGHTS_FILE_TEMPLATE.format(
                    model_name=model_to_save.__class__.__name__)
            )

            # reload the model
            if params.do_eval:
                if params.is_master:
                    logger.info('Reloading the model')
                config = DistilBertConfig.from_pretrained(
                    str(params.output_dir / TRAINED_CONFIG_FILE_TEMPLATE.format(
                        model_name=model_to_save.__class__.__name__))
                )
                model = DistilBertForQuestionAnswering.from_pretrained(
                    str(params.output_dir / TRAINED_WEIGHTS_FILE_TEMPLATE.format(
                        model_name=model_to_save.__class__.__name__)),
                    config=config
                )
                model = model.to(params.device)

    # perform the evaluation
    if params.do_eval and params.is_master:
        # initialize the training dataset
        logger.info('Initializing the evaluation dataset')
        eval_dataset, examples, features = load_and_cache_examples(
            squad_dir=params.squad_dir,
            split='dev',
            tokenizer=tokenizer,
            max_sequence_len=params.max_sequence_len,
            max_query_len=params.max_query_len,
            doc_stride=params.doc_stride,
            output_examples=True,
            overwrite_cache=params.overwrite_cache,
            is_master=params.is_master
        )

        # initialize the sampler
        logger.info('Initializing the evaluation sampler')
        eval_sampler = SequentialSampler(eval_dataset)

        # initialize the dataloader
        logger.info('Initializing the evaluation dataloader')
        eval_dataloader = DataLoader(
            dataset=eval_dataset,
            sampler=eval_sampler,
            batch_size=params.eval_batch_size
        )

        # start evaluating
        logger.info('Starting the evaluation')
        results = evaluate(
            output_dir=params.output_dir,
            model=model,
            tokenizer=tokenizer,
            max_answer_len=params.max_answer_len,
            do_lower_case=params.do_lower_case,
            dataloader=eval_dataloader,
            examples=examples,
            features=features,
            device=params.device,
            local_rank=params.local_rank,
            use_tqdm=True
        )

        # log results
        logger.info('Evaluation results:')
        for key, result in results.items():
            logger.info(f' {key}: {result}')

        # dump results
        json.dump(
            results,
            open(params.output_dir / RESULTS_FILE_TEMPLATE.format(model_name=model.__class__.__name__), 'w'),
            indent=4
        )
    
    if params.is_master:
        logger.info('Done')


if __name__ == '__main__':
    main()
