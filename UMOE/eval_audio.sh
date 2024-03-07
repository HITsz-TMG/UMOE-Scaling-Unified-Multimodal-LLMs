cd /path/to/UMOE
conda activate umoe
# data_type: video vqa clothoaqa clothov1/2
python umoe_audio/eval.py\
 --data_path /path/to/clotho.json\
 --data_type clothov1\
 --output test.json