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
    --model_path output/Uni_MoE_speech_8moe \
    --data_path path/to/eval.json \
    --enable_deepspeed_moe True \
    --data_type vqa\
    --eval_ep_size 2 \
    --mlp_dir path/to/MoeMerge\
    --version v1 \
    --vision_tower checkpoints/clip-vit-large-patch14-336 \
    --audio_tower checkpoints/whisper-small \
    --output_dir Uni_MoE_speech_eval_out.json

deepspeed --num_gpus=2 --num_nodes=1 \
    --master_addr "localhost" --master_port $MASTER_PORT \
    Uni_MoE_speech/eval.py \
    --deepspeed ./scripts/zero2.json \
    --model_base /data/lyx/jsy/Uni-MoE-speech-base \
    --model_path /data/lyx/zwq/Uni_MoE/output/Uni_MoE_speech_test_final \
    --data_path /data/share/datasets/moe/video_act/llava_activitynet_test_uni_fir.json \
    --enable_deepspeed_moe True \
    --data_type vqa\
    --eval_ep_size 2 \
    --mlp_dir /data/lyx/jsy/MoeMerge\
    --version v1 \
    --vision_tower /data/lyx/jsy/clip-vit-large-patch14-336 \
    --audio_tower /data/lyx/jsy/whisper-small \
    --output_dir Uni_MoE_speech_eval_out.json