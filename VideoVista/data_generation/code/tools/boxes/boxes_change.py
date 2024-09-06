# 对Objects进行处理
# 首先去除频率较高的一部分
# 之后去除占比过大或者过小的
import json
import cv2
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Merge")
parser.add_argument("--boxes_origin", type=str)
parser.add_argument("--boxes_save", type=str)
args = parser.parse_args()

high_frequency_words = ["man", "person", "woman", 'girl', 'child person', 'boy', "hand", 'face', "food", "table",
                        'plate', 'shirt', 'bowl', 'container', 'home appliance', 'building', 'face', 'kitchen counter',
                        'pepper', 'floor']
path = args.boxes_origin
save_path = args.boxes_save

results = {}

for filedir in tqdm(os.listdir(path)):
    # if filedir != "_QmJbWVFxm0.11":
    #     continue
    filepath = os.path.join(path, filedir)
    for json_name in os.listdir(filepath):
        second = json_name.split(".")[0]

        save_name = f"{filedir}.{second}"

        json_path = os.path.join(filepath, json_name)
        json_data = json.load(open(json_path, "r"))

        maskes = json_data["mask"]
        size = maskes[0]["size"]
        w, h = size
        total_area = size[0] * size[1]
        # remove background
        maskes = maskes[1:]
        result = []
        for mask in maskes:
            label = mask["label"]
            logit = mask["logit"]
            box = mask["box"]

            # 高频词过滤
            # if label in high_frequency_words:
            #     continue

            # logits过滤
            if logit < 0.5:
                continue

            # 区域大小过滤
            object_area = (box[2] - box[0]) * (box[3] - box[1])
            area_c = object_area / total_area
            # if area_c > 0.5:
            if area_c > 0.5 or area_c < 0.004:
                continue

            t_box = [box[0] / w, box[1] / h, box[2] / w, box[3] / h]
            t_box = [round(b, 2) for b in t_box]

            result.append({"box": t_box, "label": label, "logit": logit})

        results[save_name] = result

json.dump(results, open(save_path, "w"), indent=1)