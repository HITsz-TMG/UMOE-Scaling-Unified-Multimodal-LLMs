
conda activate unimoe_v2

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=10079
export GPUS_PER_NODE=2

cd /path/to/Uni_MoE_v2

deepspeed --num_gpus=2 --num_nodes=1 \
    --master_addr "localhost" --master_port $MASTER_PORT \
    Uni_MoE_speech/inference_new.py \
    --deepspeed ./scripts/zero2.json \
    --model_base path/to/Uni-MoE-speech-base \
    --model_path output/Uni_MoE_speech_test_final \
    --data_path /path/to/eval.json \
    --enable_deepspeed_moe True \
    --data_type vqa\
    --eval_ep_size 2 \
    --mlp_dir path/to/MoeMerge\
    --version v1 \
    --vision_tower path/to/clip-vit-large-patch14-336 \
    --audio_tower path/to/whisper-small \
    --output_dir Uni_MoE_speech_output


