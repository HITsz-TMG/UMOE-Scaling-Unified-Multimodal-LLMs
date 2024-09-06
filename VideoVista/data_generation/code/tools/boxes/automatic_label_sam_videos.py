# 针对多张图像文件进行抽取
# 提前进行判断，如果没有两到五个人的视频我们去除
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import json
import torch
import torchvision
from PIL import Image
import shutil
# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import openai
from tqdm import tqdm
import sys
# Recognize Anything Model & Tag2Text


sys.path.append('Tag2Text')
print(sys.path)


from Tag2Text.models import tag2text
from Tag2Text import inference_ram
import torchvision.transforms as TS


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(save_path, box_list, label_list, size):
    value = 0  # 0 for background
    json_data = {
        'mask': [{
            'value': value,
            'label': 'background',
            'size': size
        }]
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        box_l = box.numpy().tolist()
        box_l_r = [round(box, 4) for box in box_l]
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box_l_r,
        })
    with open(os.path.join(save_path), 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                        help="path to config file")
    parser.add_argument(
        "--ram_checkpoint", type=str, default="./Tag2Text/ram_swin_large_14m.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--grounded_checkpoint", type=str, default="groundingdino_swint_ogc.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image_dir", type=str,
                        default="processed_data/split_frames"
                        , help="path to image file")
    parser.add_argument(
        "--output_dir", "-o", type=str,
        default="processed_data/meta/split_boxes/boxes", help="output directory"
    )

    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument("--openai_key", type=str, help="key for chatgpt")
    parser.add_argument("--openai_proxy", default=None, type=str, help="proxy for chatgpt")

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    ram_checkpoint = args.ram_checkpoint  # change the path of the model
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path_dir = args.input_image_dir
    split = args.split
    openai_key = args.openai_key
    openai_proxy = args.openai_proxy
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    # initialize Recognize Anything Model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
        TS.Resize((384, 384)),
        TS.ToTensor(), normalize
    ])
    # load model
    ram_model = tag2text.ram(pretrained=ram_checkpoint,
                             image_size=384,
                             vit='swin_l')
    ram_model.eval()
    ram_model = ram_model.to(device)

    # initialize SAM
    if use_sam_hq:
        print("Initialize SAM-HQ Predictor")
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_list = list(os.listdir(image_path_dir))

    for video_name in tqdm(image_list):
        video_path = os.path.join(image_path_dir, video_name)

        flag = 0
        for idx, image_name in enumerate(os.listdir(video_path)):

            image_path = os.path.join(video_path, image_name)
            json_name = image_name.split("s.")[0] + ".json"

            save_dir = os.path.join(output_dir, video_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_path = os.path.join(output_dir, video_name, json_name)

            if os.path.exists(save_path):
                continue

            try:
                image_pil, image = load_image(image_path)
            except:
                print(f"{image_name} is not a image!")
                continue
            raw_image = image_pil.resize((384, 384))
            raw_image = transform(raw_image).unsqueeze(0).to(device)

            res = inference_ram.inference(raw_image, ram_model)

            tags = res[0].replace(' |', ',')

            boxes_filt, scores, pred_phrases = get_grounding_output(
                model, image, tags, box_threshold, text_threshold, device=device
            )

            fine_boxes_filt = boxes_filt.clone()
            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
            boxes_filt = boxes_filt[nms_idx]
            fine_boxes_filt = fine_boxes_filt[nms_idx]
            pred_phrases = [pred_phrases[idx] for idx in nms_idx]

            save_mask_data(save_path, boxes_filt, pred_phrases, size)
