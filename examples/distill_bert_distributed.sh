#!/bin/bash
export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=4
export WORLD_SIZE=4

python -m torch.distributed.launch \
    --nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    distill_bert_distributed.py \
        --data_file path/to/data/file.txt \
        --output_dir path/to/output/dir \
        --force \
        --student_config_file path/to/student/config/file.json \
        --student_weights_file path/to/student/weights/file.path \
        --teacher_type bert-base-uncased \
        --tokenizer_vocab_file path/to/tokenizer/vocab/file.txt \
        --do_tokenize \
        --do_lower_case \
        --seed 42 \
        --use_cuda \
        --use_distributed
