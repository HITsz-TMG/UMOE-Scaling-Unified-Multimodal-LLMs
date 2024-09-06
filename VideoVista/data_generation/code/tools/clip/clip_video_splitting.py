import os
import json
import torch
from tqdm import tqdm
from PIL import Image
import argparse
from transformers import CLIPModel, CLIPImageProcessor

parser = argparse.ArgumentParser(description="Merge")
parser.add_argument("--input_dir", type=str,
                    default="/data/cxy/VideoLLaVA/tools/dataset/panda_final/data/frames/Film & Animation")
parser.add_argument("--save_path", type=str,
                    default="/data/cxy/VideoLLaVA/tools/dataset/panda_pool/infos/splitting_points.json")
args = parser.parse_args()

model = CLIPModel.from_pretrained("/data/share/Model/clip-vit-large-patch14")
model = model.cuda().bfloat16().eval()
processor = CLIPImageProcessor.from_pretrained("/data/share/Model/clip-vit-large-patch14")

dir = args.input_dir
save_path = args.save_path

if os.path.exists(save_path):
    with open(save_path, 'r') as f:
        existing_splitting_dict = {list(json.loads(line).keys())[0]: list(json.loads(line).values())[0] for line in f}
else:
    existing_splitting_dict = {}

splitting_dict = {}

# 遍历视频文件夹
for filename in tqdm(os.listdir(dir)):
    print(filename)
    # 检查是否已经处理过
    if filename in existing_splitting_dict:
        print(f"{filename} has already been processed")
        continue

    filepath = os.path.join(dir, filename)
    frame_count = len(list(os.listdir(filepath)))

    if frame_count > 40:
        image_feats = []

        # 提取特征
        with torch.no_grad():
            for image_name in os.listdir(filepath):
                image_path = os.path.join(filepath, image_name)
                img = Image.open(image_path)
                image = processor.preprocess(img, return_tensors="pt")["pixel_values"].cuda().bfloat16()
                clip_image_features = model.get_image_features(image)
                image_feats.append(clip_image_features)

        # 计算相似度
        scores = []
        for idx in range(len(image_feats) - 1):
            feat1 = image_feats[idx]
            feat2 = image_feats[idx + 1]
            feat1 /= feat1.norm(dim=-1, keepdim=True)
            feat2 /= feat2.norm(dim=-1, keepdim=True)
            score = torch.nn.functional.cosine_similarity(feat1, feat2).item()
            scores.append(score)

        # 获取拆分点
        begin = 0
        splitting_points = []
        while begin + 39 <= len(image_feats):
            current_scores = scores[begin:begin + 39]
            score_index = current_scores.index(min(current_scores))
            begin += score_index + 1
            splitting_points.append(begin - 1)
        splitting_points.append(len(image_feats))
        splitting_dict[filename] = splitting_points

        # 将结果写入json文件
        with open(save_path, 'a') as f:
            f.write(json.dumps({filename: splitting_points}) + "\n")

print(f"Processed {len(splitting_dict)} new videos.")
