# 增加补充信息
"""
使用gpt来帮助生成动作识别
输入：存放原始视频的frames的文件夹（因为前面做了切分的原因，现在frame可能）
输出：json文件
"""

import os
# import openai
from openai import OpenAI
import warnings
from PIL import Image, ImageDraw
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse
import base64
import json
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')
client = OpenAI(
    base_url="", api_key=""
)
model = "gpt-4-vision-preview"



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

parser = argparse.ArgumentParser(description="action recongnition")
parser.add_argument("--audios", type=str)
parser.add_argument("--frames", type=str)
parser.add_argument("--save", type=str)
parser.add_argument("--error", type=str)
# 添加一个日志 // 如果出现是gpt调用后 回答有问题的 / 需要记录在这个文件之中 /
parser.add_argument("--video_titles", type=str)
parser.add_argument("--video_categorys", type=str)


args = parser.parse_args()

video_titles = json.load(open(args.video_titles, "r"))
video_categorys = json.load(open(args.video_categorys, "r"))

## 这里我需要把这个路径传输到变量名


video_audios = json.load(open(args.audios, "r"))
# 我输入的视频帧文件路径 已经一次过滤掉错误的视频了 // 所以此处没有指定error
try:
    error_videos = json.load(open(args.error, "r"))
except:
    error_videos = []

base_dir = args.frames
base_save_path = args.save


def log_error(video_name, error_message):
    with open(args.error_log, "a") as f:
        f.write(f"{video_name}: {error_message}\n")


def gpt_forward(frame_dir_name):
    global cnt0
    frame_dir = os.path.join(base_dir, frame_dir_name)
    save_path = os.path.join(base_save_path, f"{frame_dir_name}.json")
    print(save_path)
    #判断处理完成的视频 // 跳过处理
    if os.path.exists(save_path) and frame_dir_name not in error_videos:
        return None
    #根据frames的数量判断秒数
    video_length = len(os.listdir(frame_dir))
    if video_length == 0 or video_length > 40:
        progress_bar.update(1)
        return None

    # 每一个视频对应的帧文件夹
    images_list = []
    # frame_name 就是 1s.jpg / 2s.jpg ...
    for frame_name in os.listdir(frame_dir):
        second = int(frame_name.split("s.")[0])
        frame_path = os.path.join(frame_dir, frame_name)
        frame_image = Image.open(frame_path)
        images_list.append((second, frame_image))

    # 根据秒数 进行排序
    images_list = sorted(images_list, key=lambda x: x[0])

    images_list = [tup[1] for tup in images_list]

    base64_images = []

    # 视频秒数 小于15s的话 / 3张图片做水平拼接 / 并且还保存了图片
    if len(images_list) <= 15:
        frame_each_image = "three"
        number_second = 3
        for idx in range(0, len(images_list), 3):
            current_images = images_list[idx:idx + 3]
            merged_image = merge_images_horizontally(current_images, spacing=30)
            buffered = BytesIO()
            merged_image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(base64_image)

    else:
        frame_each_image = "four"
        number_second = 4
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
    # print(audio_text)

    # Title
    title_text = video_titles[frame_dir_name.split(".")[0]]
    category_text = video_categorys[frame_dir_name.split(".")[0]][0]

    # print(title_text)
    contents.append(
        {
            "type": "text",
            "text": f"The input video contains a total of {len(images_list) - 1} seconds, so the number of frames in the input is {len(images_list)}.\n"
                    f"Video Title: {title_text}\n"
                    f"Video Category: {category_text}\n"
                    f"Audio Transcripts: {audio_text}\n"
        }
    )

    # message提到的信息：我要的信息就是 task instruction里面提到的

    messages = [
        {
            "role": "system",
            "content":
                "You are an AI visual assistant specialized in analyzing videos.\n"
                "You will receive a series of composed images and each composite image contains temporal frames. All composed images form a clip of a video. Additionally, we provide you with some additional information about the video: Title, Category, and Audio transcripts.\n"
                "#For the video frames:\n"
                f"1. Except for the last image, each of the composed images contains {frame_each_image} temporal frames from the video from left to right. The number of temporal frames contained in the last image is less than or equal to {frame_each_image}.\n"
                f"2. The first frame of first image corresponds to the 0 second of the video, the second frame of first image corresponds to the 1 second of the video, while the first frame of second image corresponds to the {number_second} second of the video, and so on.\n"
                "#For the Title:\n"
                "The video title provides insight into the video's main theme but shouldn't be solely relied upon for action information.\n"
                "# For the Category:\n"
                "The video category is self-labeled by users upon upload, hence errors may exist.\n"
                "#For the Audio Transcripts:\n"
                "1. The audio transcript may be none, and its format is starting with 'Second {begin_second} to {end_second}'.\n"
                "2. The audio transcript may be noisy. When using it, you need to judge for yourself whether the information in it is valid.\n"
                "3. The actions mentioned in the audio transcript may not be the actions being performed by the speaker.\n"
                "-----\n"
                "##Task Instruction:\n"
                "Your task is to recognize all the actions in the provided video along with the subjects corresponding to the actions, and segment them based on time.\n"
                "1. The 'Time' consists of two parts: start second and end second.\n"
                "2. The 'Subject' should include prominent features such as gender, clothing, etc., and be expressed as an adjective phrase. If possible, the subject is preferably a person.\n"
                "3. The 'Action' should be as detailed as possible. You don't need to break down a single action into multiple smaller actions; instead, output a detailed and complete action.\n"
                "4. If there are no obvious actions in the image, use 'none' to represent it. Therefore, you need to carefully analyze the input frames to make a judgment.\n"
                "5. When multiple subjects are executing actions simultaneously, you need to provide multiple recognition results for the corresponding time periods. Just like the last example in the Reference Examples.\n"
                "Based on the above demands, generate the response in the form of a Python list of dictionary strings with keys \"Time\" for the time segment, \"Subject\" for the subject of the action and \"Action\" for the recognized action.\n"
                "The final output format is: [{\"Time\": [begin_second, end_second], \"Subject\": \"Subject of Action\", \"Action\": \"Your recognized action\"}, ..., {\"Time\": [begin_second, end_second], \"Subject\": \"Subject of Action\", \"Action\": \"Your recognized action\"}]\n"
                "##Reference Examples:\n"
                "[{\"Time\": [0, 3], \"Subject\": \"Woman athletes wearing different colored clothes\", \"Action\": \"Running on the sports field\"}, {\"Time\": [3, 5], \"Subject\": \"A dense crowd of spectators in the stands\", \"Action\": \"Cheering for the athletes\"}]\n"
                "[{\"Time\": [0, 9], \"Subject\": \"A man wearing a red shirt\", \"Action\": \"Using a fork to dig food out of a shell\"}]\n"
                "[{\"Time\": [0, 1], \"Subject\": \"Woman with mustard sleeve and green apron\", \"Action\": \"Using a screwdriver to loosen the screws on the kitchen cupboard doors\"}, {\"Time\": [1, 3], \"Subject\": \"Woman with mustard sleeve and green apron\", \"Action\": \"Removing the loosened screws from the kitchen cupboard doors by hand\"}]\n"
                "[{\"Time\": [0, 18], \"Subject\": \"Black Volvo car with 'AUTONOMOUS PARKING TEST VEHICLE'\", \"Action\": \"Reversing into a parking spot\"}]\n"
                "[{\"Time\": [0, 3], \"Subject\": \"none\", \"Action\": \"none\"}]\n"
                "[{\"Time\": [0, 5], \"Subject\": \"Man in a black t-shirt and woman in a red hoodie\", \"Action\": \"Walking down a rural road\"}, {\"Time\": [0, 2], \"Subject\": \"A woman wearing black pants and a black-and-white top\", \"Action\": \"Walking backwards on a rural road\"}, {\"Time\": [3, 5], \"Subject\": \"A woman wearing black pants and a black-and-white top\", \"Action\": \"Turn around and run to chase the brownish-yellow dog\"}, {\"Time\": [1, 5], \"Subject\": \"A brownish-yellow dog\", \"Action\": \"Running down a rural road\"}]\n"
        },
        {
            "role": "user",
            "content": contents
        }
    ]
    # print(messages[0]["content"])

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
        )
        response_message_0 = response.choices[0].message.content
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


def save_results(results, output_file):
    json.dump(results, open(output_file, "w"))

os.makedirs(base_save_path, exist_ok=True)

# 当前进程数设置为 1
with ThreadPoolExecutor(max_workers=1) as executor:
    futures = []
    file_list = list(os.listdir(base_dir))
    file_list = sorted(file_list)
    total_dirs = len(file_list)
    print(total_dirs)

    global progress_bar
    progress_bar = tqdm(total=total_dirs, desc="Processing")

    for frame_dir_name in file_list:
        futures.append(executor.submit(gpt_forward, frame_dir_name))

        # gpt_forward(frame_dir_name)

# progress_bar.close()
