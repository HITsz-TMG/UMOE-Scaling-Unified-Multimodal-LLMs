conda activate unimoe_v2

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=10079
export GPUS_PER_NODE=2

cd /path/to/Uni_MoE_v2

deepspeed --num_nodes 1 --num_gpus 8 \
    --master_addr "localhost" \
    --master_port $MASTER_PORT \
    Uni_MoE_speech/train/train_mem_speech.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /path/to/Uni-MoE-speech-base  \
    --version v1 \
    --data_path /path/to/data_with_infos.json \
    --vision_tower /path/to/clip-vit-large-patch14-336 \
    --audio_tower /path/to/whisper-small \
    --mm_projector_type mlp2x_gelu\
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir output/Uni_MoE_speech_8moe \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
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
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to none \
    --tune_mm_mlp_adapter True \
    --pretrain_mm_mlp_adapter /path/to/Uni-MoE-speech-base/mm_projector.bin \
    --tune_mm_audio_projector True \
    --pretrain_audio_aligner /path/to/Uni-MoE-speech-base/mm_audio_aligner.bin \
    --llm_lora_enable True \
    --mix_va True \
    --enable_deepspeed_moe True \
    --group_by_modality_length True \
    --mlp_dir /path/to/Uni-MoE-v2-Experts
