#!/bin/bash

# This is the training script for the UMOE, only for MoE-training stage.

# Uncomment and set the following variables correspondingly to run this script:

conda activate umoe
PROMPT_VERSION=v1
export MASTER_PORT=9873
cd /path/to/UMOE

deepspeed --num_gpus 1 --num_nodes 1\
    --master_addr "localhost" --master_port $MASTER_PORT \
    umoe_audio/train/train_mem_audio.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path checkpoints/UMOE-audio-base \
    --version $PROMPT_VERSION \
    --data_path /path/to/cap_all.json \
    --image_folder /path/to/train2014.zip \
    --vision_tower checkpoints/clip-vit-large-patch14-336 \
    --audio_tower checkpoints/BEATs_iter3_plus_AS2M.pt \
    --mm_projector_type mlp2x_gelu\
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --output_dir output/UMOE_audio_ckpt \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate 4e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 3 \
    --lazy_preprocess True \
    --report_to none \
    --tune_mm_mlp_adapter True\
    --pretrain_mm_mlp_adapter checkpoints/UMOE-audio-base/mm_projector.bin\
    --tune_mm_audio_projector True\
    --pretrain_audio_aligner checkpoints/UMOE-audio-base/mm_audio_aligner.bin\
    --llm_lora_enable True \
    --mix_va True
