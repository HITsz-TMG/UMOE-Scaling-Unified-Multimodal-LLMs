# 事件级别
# 事件是更高一级的action
# 重点在于对视频的整体描述

import os
from openai import OpenAI
import warnings
from PIL import Image, ImageDraw
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import base64
import json
import argparse
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')
client = OpenAI(
    base_url="", api_key=""
)
model = "gpt-4-vision-preview"

def log_error(video_name, error_message):
    with open(args.error_log, "a") as f:
        f.write(f"{video_name}: {error_message}\n")


def merge_images_horizontally(image_list, spacing=10):
    # 获取第一张图像的尺寸
    width, height = image_list[0].size

    total_width = sum(img.width for img in image_list) + spacing * (len(image_list) - 1)
    # 创建一个新的图像，宽度为所有图像的总宽度，高度为第一张图像的高度
    merged_image = Image.new('RGB', (total_width, height))

    # 将图像逐个水平拼接到新图像上
    x_offset = 0
    for img in image_list:
        merged_image.paste(img, (x_offset, 0))

        x_offset += img.width + spacing

    return merged_image


parser = argparse.ArgumentParser(description="event recognition")

parser.add_argument("--audios", type=str)
parser.add_argument("--frames", type=str)
parser.add_argument("--save", type=str)
parser.add_argument("--boxes", type=str)
parser.add_argument("--action", type=str)
parser.add_argument("--error", type=str)
parser.add_argument("--video_titles", type=str)
parser.add_argument("--video_categorys", type=str)

args = parser.parse_args()

video_titles = json.load(open(args.video_titles, "r"))
video_categorys = json.load(open(args.video_categorys, "r"))

video_audios = json.load(open(args.audios, "r"))
video_bounding_boxes = json.load(open(args.boxes, "r"))
try:
    error_videos = json.load(open(args.error, "r"))
except:
    error_videos = []
base_dir = args.frames
base_save_path = args.save
cnt0 = 0


# /cnt40 = 0
def gpt_forward(frame_dir_name):
    global cnt0
    frame_dir = os.path.join(base_dir, frame_dir_name)
    save_path = os.path.join(base_save_path, f"{frame_dir_name}.json")

    if os.path.exists(save_path) and frame_dir_name not in error_videos:
        return None

    video_length = len(os.listdir(frame_dir))
    if video_length == 0 or video_length > 40:
        progress_bar.update(1)
        return None

    images_list = []
    for frame_name in os.listdir(frame_dir):
        second = int(frame_name.split("s.")[0])
        frame_path = os.path.join(frame_dir, frame_name)
        frame_image = Image.open(frame_path)
        images_list.append((second, frame_image))

    images_list = sorted(images_list, key=lambda x: x[0])
    images_list = [tup[1] for tup in images_list]
    # 每四张拼接成一张图像

    base64_images = []
    if len(images_list) <= 15:
        number_second = 3
        frame_each_image = "three"
        for idx in range(0, len(images_list), 3):
            current_images = images_list[idx:idx + 3]
            merged_image = merge_images_horizontally(current_images, spacing=30)
            buffered = BytesIO()
            merged_image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(base64_image)

    else:
        number_second = 4
        frame_each_image = "four"
        for idx in range(0, len(images_list), 4):
            current_images = images_list[idx:idx + 4]
            merged_image = merge_images_horizontally(current_images, spacing=30)
            buffered = BytesIO()
            merged_image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(base64_image)

    contents = []
    for base64_image in base64_images:
        contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

    # Audio
    audio_text = video_audios[frame_dir_name]['audio_transcript']
    if audio_text == '':
        audio_text = 'none'


    title_text = video_titles[frame_dir_name.split(".")[0]]
    category_text = video_categorys[frame_dir_name.split(".")[0]][0]
    # boxes
    boxes_text = ""
    for second in range(0, len(images_list)):
        box_id = f"{frame_dir_name}.{second}"
        boxes = video_bounding_boxes[box_id]
        boxes_text += f"Second {second}: "
        for box in boxes:
            boxes_text += f"{box['label']}: {box['box']}"
            print(boxes_text)
            if box != boxes[-1]:
                boxes_text += ", "

        boxes_text = boxes_text + "\n"
    boxes_text = boxes_text.strip()


    print("1212", boxes_text)

    # action
    action_path = os.path.join(args.action, f"{frame_dir_name}.json")
    try:
        Action = json.load(open(action_path, "r"))["result"]
    except:
        Action = "none"

    # print(Action)
    if "error" in Action.lower() or "time" in Action.lower():
        Action = "none"

    print(Action)

    contents.append(
        {
            "type": "text",
            "text": f"The input video contains a total of {len(images_list) - 1} seconds, so the number of frames in the input is {len(images_list)}.\n"
                    f"Video Title: {title_text}\n"
                    f"Video Category: {category_text}\n"
                    f"Audio Transcripts: {audio_text}\n"
                    f"Boxes: {boxes_text}\n"
                    f"Action: {Action}\n"
        }
    )

    messages = [
        {
            "role": "system",
            "content":
                "You are an AI visual assistant specialized in analyzing videos.\n"
                "You will receive a series of composed images and each composite image contains temporal frames. All composed images form a clip of a video. Additionally, we provide you with some additional information about the video: Action Information, Title, Category, Audio transcripts and Boxes.\n"
                "#For the video frames:\n"
                f"1. Except for the last image, each of the preceding images contains {frame_each_image} temporal frames from the video from left to right. The number of temporal frames contained in the last image is less than or equal to {frame_each_image}.\n"
                f"2. The first frame of input corresponds to the 0 second of the video, while the second frame corresponds to the 1 second of the input video, while the first frame of second image corresponds to the {number_second} second of the video, and so on.\n"
                "#For the Action Information:\n"
                "1. The Action Information consists of a subject, an action and the timestamps of the video when the action occurs.\n"
                "2. The Action Information may be error, before use, you need to judge it in combination with the input frame images.\n"
                "# For the Title:\n"
                "The video title provides insight into the video's main theme but shouldn't be solely relied upon for action information.\n"
                "#For the Category:\n"
                "The video category is self-labeled by users upon upload, hence errors may exist.\n"
                "#For the Audio Transcripts:\n"
                "1. The audio transcript may be none, and its format is starting with 'Second {begin_second} to {end_second}'.\n"
                "2. The audio transcript may be noisy. When using it, you need to judge for yourself whether the information in it is valid.\n"
                "#For the Boxes:\n"
                "1. For the video frame at the {second} second of input, we provide corresponding the boxes, starting with 'Second {second}'.\n"
                "2. The boxes for each second is provided in the form of '{label}: {box}', and if there are multiple boxes, they are separated by ','.\n"
                "3. The label within the boxes may be error; you need to judge based on the image when using it.\n"
                "-----\n"
                "##Task Instruction:\n"
                "Your task is to describe the input video into one specific event, each part of the event requiring a temporal relationship. The description of characters in the event should include observable information such as clothing and gender.  Note the temporal action change and the main content this video presents.\n"
                "When using the information from the frame and audio transcript, directly explain the scene, and do not mention that the information source is the frame or audio transcript. Always answer as if you are directly looking at the video.\n"
                "In all the input information, the images formed by frames are the decisive information, while all other information serves as auxiliary information to help you understand the video.\n"
                "Please generate the response in the form of a Python list of dictionary strings with keys \"Event\" for the event you detected, same as: [{\"Event\": \"your detected event.\"}]\n"
                "##Reference Examples:"
                "[{\"Event\": \"A man wearing a red shirt uses a fork to dig food out of a shell. Subsequently, he places the retrieved food into his mouth and begins to chew.\"}]\n"
                "[{\"Event\": \"The person wearing a white shirt first pours an ounce of simple syrup into a small glass. Then, he pours the liquid from the small glass into a large glass.\"}]\n"
                "[{\"Event\": \"A man in a business suit stands at a busy urban intersection, checking his watch. He waves down a yellow taxi, which pulls up beside him. He opens the door, exchanges a few words with the driver, and climbs into the backseat of the cab.\"}]\n"
        },
        {
            "role": "user",
            "content": contents
        }
    ]

    try:
        completion_0 = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            # max_tokens=2048
        )
        response_message_0 = completion_0.choices[0].message.content
        print(f"{frame_dir_name}: {response_message_0}")
        result = {
            "video_name": frame_dir_name,
            "result": response_message_0
        }

    except Exception as e:
        print(f"Video: {frame_dir_name}, Error: {e}")
        result = {
            "video_name": frame_dir_name,
            "result": str(e)
        }

    json.dump(result, open(save_path, "w"))
    progress_bar.update(1)


os.makedirs(base_save_path, exist_ok=True)

# 当前进程数设置为1r
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = []

    file_list = list(os.listdir(base_dir))
    file_list = sorted(file_list)
    # file_list = file_list[:10]
    total_dirs = len(file_list)
    global progress_bar
    progress_bar = tqdm(total=total_dirs, desc="Processing")

    for frame_dir_name in file_list:
        futures.append(executor.submit(gpt_forward, frame_dir_name))
        # gpt_forward(frame_dir_name)

