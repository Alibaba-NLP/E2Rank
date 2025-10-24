#!/bin/bash

set -e

export FORCE_TORCHRUN=1 
export NNODES=1
export NODE_RANK=8


model_name_or_path=Alibaba-NLP/E2Rank-0.6B-Embedding-Only
data_path=data/train.jsonl
model_name=E2rank-0.6B


torchrun \
  --nnodes=$NNODES --nproc_per_node=$NODE_RANK \
  src/run.py \
  --deepspeed ./scripts/zero3.json \
  --output_dir checkpoints/$model_name \
  --model_name_or_path $model_name_or_path \
  --use_embed_loss True \
  --use_ranknet_loss True \
  --loss_ranknet_factor 2.0 \
  --ranknet_scale_factor 10.0 \
  --data_path $data_path \
  --bf16 \
  --tf32 True \
  --per_device_train_batch_size 8 \
  --gradient_checkpointing \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-6 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "linear" \
  --num_train_epochs 1 \
  --logging_strategy "steps" \
  --logging_steps 1 \
  --save_strategy "steps" \
  --save_steps 200 \
  --overwrite_output_dir

