cd /path/to/Uni_MoE
conda activate unimoe
# data_type: video vqa clothoaqa clothov1/2
python Uni_MoE_audio/eval.py\
 --data_path /path/to/clotho.json\
 --data_type clothov1\
 --output test.json