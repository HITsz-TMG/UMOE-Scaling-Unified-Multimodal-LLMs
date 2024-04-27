
export MASTER_ADDR="localhost"
export MASTER_PORT=9993
export GPUS_PER_NODE=8
conda activate unimoe

cd path/to/Uni_MoE

deepspeed  --num_gpus 2 --num_nodes 1\
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    Uni_MoE_speech_dp/train/train_mem_speech.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path path/to/speech_base \
    --version v1 \
    --data_path path/to/train.json \
    --image_folder /path/to/train2014.zip \
    --vision_tower path/to/clip-vit-large-patch14-336 \
    --audio_tower path/to/whisper-small \
    --mm_projector_type mlp2x_gelu\
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir output/Uni_MoE_speech_dp \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
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
    --pretrain_mm_mlp_adapter path/to/mm_projector.bin\
    --tune_mm_audio_projector True\
    --pretrain_audio_aligner path/to/mm_audio_aligner.bin\
    --llm_lora_enable True \
    --mix_va True \
    --group_by_modality_length True