#!/bin/bash



echo "----------------------- WANDB --------------------------"

cd /path to/DC_MoE/github

export WANDB_API_KEY='your Weights and Biases api key'
export WANDB_PROJECT='Dynamic-Capacity_MoEV5_Qwen2VL_2B'

# **WANDB_WATCH** (`str`, *optional* defaults to `"false"`):
# Can be `"gradients"`, `"all"`, `"parameters"`, or `"false"`. Set to `"all"` to log gradients and
# parameters.
export WANDB_WATCH='false'

# WANDB_LOG_MODEL** (`str`, *optional*, defaults to `"false"`):
# Whether to log model and checkpoints during training. Can be `"end"`, `"checkpoint"` or `"false"`. If set
# to `"end"`, the model will be uploaded at the end of training. If set to `"checkpoint"`, the checkpoint
# will be uploaded every `args.save_steps` . If set to `"false"`, the model will not be uploaded. Use along
# with [`~transformers.TrainingArguments.load_best_model_at_end`] to upload best model.
export WANDB_LOG_MODEL='false'

time=$(date "+%m-%d-%H-%M")

export NAME="DC_MoE-training"

export SAVE_PATH="./outputs" 

export WANDB_MODE="offline"

MASTER_PORT=9226

ASCEND_LAUNCH_BLOCKING=1 deepspeed \
    --master_addr "localhost" \
    --master_port $MASTER_PORT \
    ./training/train_unimoev2_qwen2vl.py \
    --moe_copy all \
    --attn_implementation sdpa \
    --deepspeed ./training/deepspeed_zero2.conf \
    --initialize True \
    --model_name_or_path Qwen2-VL-2B-Instruct \
    --processor_path Qwen2-VL-2B-Instruct \
    --data_path ./data \
    --image_root ./image \
    --fp32_gate True \
    --ep_size 1 \
    --dynamic_mlp_size_factor 4 \
    --fixed_mlp_size_factor 8 \
    --mlp_dynamic_expert_num 4 \
    --mlp_dynamic_top_p 0.7 \
    --mlp_dynamic_top_k 0 \
    --mlp_fixed_expert_num 2 \
    --mlp_dynamic_null_expert_num 1 \
    --token_drop True \
    --drop_policy "probs" \
    --min_capacity 8 \
    --capacity_factor 3 \
    --aux_balance_weight 10 \
    --l_aux_weight 0.025 \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.00 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --report_to "none" \
    --run_name "${NAME}_${time}"
