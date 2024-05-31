conda activate unimoe_v2

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=10029
export GPUS_PER_NODE=2

cd path/to/Uni_MoE_v2

deepspeed --num_gpus=2 --num_nodes=1 \
    --master_addr "localhost" --master_port $MASTER_PORT \
    Uni_MoE_speech/eval.py \
    --deepspeed ./scripts/zero2.json \
    --model_base checkpoints/Uni-MoE-speech-base \
    --model_path output/Uni_MoE_v2_e2 \
    --data_path path/to/eval.json \
    --enable_deepspeed_moe True \
    --data_type vqa\
    --eval_ep_size 2 \
    --mlp_dir path/to/Uni-MoE-v2-Experts\
    --version v1 \
    --vision_tower checkpoints/clip-vit-large-patch14-336 \
    --audio_tower checkpoints/whisper-small \
    --output_dir Uni_MoE_speech_eval_out.json
