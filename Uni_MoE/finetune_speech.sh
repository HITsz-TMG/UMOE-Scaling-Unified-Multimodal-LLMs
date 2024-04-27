#!/bin/bash

# This is the training script for the Uni-MoE, only for MoE-training stage.

# Uncomment and set the following variables correspondingly to run this script:


conda activate unimoe
PROMPT_VERSION=v1
export MASTER_PORT=9811
cd /path/to/Uni_MoE

deepspeed --num_gpus 1 --num_nodes 1\
    --master_addr "localhost" --master_port $MASTER_PORT \
    Uni_MoE_speech/train/train_mem_speech.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path checkpoints/Uni-MoE-speech-base \
    --version $PROMPT_VERSION \
    --data_path /path/to/spc_all.json \
    --image_folder /path/to/train2014.zip \
    --vision_tower checkpoints/clip-vit-large-patch14-336 \
    --audio_tower checkpoints/whisper-small \
    --mm_projector_type mlp2x_gelu\
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir output/Uni_MoE_speech_ckpt \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
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
    --pretrain_mm_mlp_adapter checkpoints/Uni-MoE-speech-base/mm_projector.bin\
    --tune_mm_audio_projector True\
    --pretrain_audio_aligner checkpoints/Uni-MoE-speech-base/mm_audio_aligner.bin\
    --llm_lora_enable True \
    --mix_va True \
    --group_by_modality_length True
