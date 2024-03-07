cd /path/to/UMOE
conda activate umoe
# data_type: video vqa mmbench
python umoe_speech/eval.py\
 --data_path /path/to/vqa_eval.json\
 --data_type vqa\
 --output test.json