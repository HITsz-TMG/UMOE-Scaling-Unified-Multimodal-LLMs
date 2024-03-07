cd /path/to/Uni_MoE
conda activate unimoe
# data_type: video vqa mmbench
python Uni_MoE_speech/eval.py\
 --data_path /path/to/vqa_eval.json\
 --data_type vqa\
 --output test.json