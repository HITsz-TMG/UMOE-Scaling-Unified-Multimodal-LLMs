# 目标检测相关的问题# Required Libraries

# 直接构造问题
# 输入图像、objects辅助信息、Title、Category

# "##Example:\n"
# "Does [object] appear in this video?"                ## 存在
# "Hom many [object] appear in this video?"            ## 数量
# "At what second does [object] appear in the video?"  ## 时间定位 出现 / 消失
# "Where does [object] appear in the video?"           ## 空间定位
# "In which direction is [object] moving?"             ## 空间tracking
# ""                                                   ## 关系(包括人物，相邻是什么) 空间关系、时序关系（谁先出现）

# Objects Existence
# Objects Number
#


import os
import re

from openai import OpenAI
import warnings
from PIL import Image, ImageDraw
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import base64
import json
import argparse
from ast import literal_eval
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


parser = argparse.ArgumentParser(description="object recognition")
parser.add_argument("--audios", type=str)
parser.add_argument("--frames", type=str)
parser.add_argument("--save", type=str)
parser.add_argument("--error", type=str)
parser.add_argument("--video_titles", type=str)
parser.add_argument("--video_categorys", type=str)

args = parser.parse_args()

video_titles = json.load(open(args.video_titles, "r"))
video_categorys = json.load(open(args.video_categorys, "r"))
video_audios = json.load(open(args.audios, "r"))
# video_bounding_boxes = json.load(open(args.boxes, "r"))
try:
    error_videos = json.load(open(args.error, "r"))
except:
    error_videos = []
base_dir = args.frames
base_save_path = args.save

# print(len(error_videos))


def gpt_forward(frame_dir_name):
    global Counterfactual_Reasoning_Count, Commonsense_Reasoning_Count, Causal_Reasoning_Count
    frame_dir = os.path.join(base_dir, frame_dir_name)

    save_path = os.path.join(base_save_path, f"{frame_dir_name}")

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


    # Title
    title_text = video_titles[frame_dir_name.split(".")[0]]
    # category
    category_text = video_categorys[frame_dir_name.split(".")[0]][0]

    contents.append(
        {
            "type": "text",
            "text": f"The input video contains a total of {len(images_list) - 1} seconds, so the number of frames in the input is {len(images_list)}.\n"
                    f"Video Title: {title_text}\n"
                    f"Video Category: {category_text}\n"
                    f"Audio Transcripts: {audio_text}\n"
        }
    )

    messages = [
        {
            "role": "system",
            "content":
                "You are an AI visual assistant specialized in analyzing videos.\n"
                "You will receive a series of composed images and each composed image contains temporal frames. All composed images form a clip of a video. Additionally, we provide you with some additional information about the video: Title, Category and Audio Transcript.\n"
                "#For the video frames:\n"
                f"1. Except for the last image, each of the preceding images contains {frame_each_image} temporal frames from the video from left to right. The number of temporal frames contained in the last image is less than or equal to {frame_each_image}.\n"
                f"2. The total seconds are equal to the total number of frames. The first frame of first image corresponds to the 0 second of the video, the second frame of first image corresponds to the 1 second of the video, while the first frame of ith image corresponds to the i*{number_second} second of the video, and so on.\n"
                "# For the Title:\n"
                "The video title provides insight into the video's main theme but shouldn't be solely relied upon for action information."
                "# For the Category:\n"
                "The video category is self-labeled by users upon upload, hence errors may exist.\n"
                "#For the Audio Transcripts:\n"
                "1. The audio transcript may be none, and its format is starting with 'Second {begin_second} to {end_second}'.\n"
                "2. The audio transcript may be noisy. When using it, you need to judge for yourself whether the information in it is valid.\n"
                "-----\n"
                "##Task Types:\n"
                "1. Counterfactual Reasoning. This type of question primarily focuses on considering what would have happened if a certain event or circumstance had been different. These question typically begins with phrases like 'What if...' or 'Imagine if...'.\n"
                "2. Commonsense Reasoning. This type of question typically requires you to apply everyday knowledge about the physical and social world to understand and reason about the video content. For example, asking whether the architecture appearing in the video is Gothic or Baroque, inquiring about the phylum of animal, inquiring about the founder of specific brand, inquiring about the sports position of athlete, asking about the material of a specific object, etc.\n"  # such as 丰富"
                "3. Causal Reasoning. This type of question mainly focuses on inferring the causal relationship between events. For instance, determining why a particular action occurred based on preceding events. These question typically begins with words like 'Why' or 'How'.\n"
                "##Task Instruction:\n"
                "Your task is to use the provided information, create one question about the video, and provide corresponding answers.\n"
                f"We have currently generated {Counterfactual_Reasoning_Count} samples of Counterfactual Reasoning, {Commonsense_Reasoning_Count} samples of Commonsense Reasoning, and {Causal_Reasoning_Count} samples of Causal Reasoning. You need to balance the quantity of samples across the three types\n"
                "When generating question, check the three task types above, select one that matches the input video, and then generate a complex question related to the video based on that task type, along with its corresponding answer.\n"
                "It's important to note that both the questions and answers you provide need to be as accurate as possible.\n"
                "When using the information from the frames and audio transcript, directly explain the scene, and do not mention that the information source is the frames and audio transcript. Always answer as if you are directly looking at the video.\n"
                "When referring to characters in the question, it is necessary to add certain adjectives, such as clothing, etc.\n"
                "In all the input information, the images formed by frames are the decisive information, while all other information serves as auxiliary information to help you understand the video.\n"
                "Please generate the response in the form of a Python dictionary string with keys \"Q\" for question, \"A\" for answer and \"Type\" for question type. Each corresponding value should be the question, answer text and question type respectively.\n"
                "For example, your response should look like this: {\"Q\": \"Your first question here...\", \"A\": \"Your first answer here\", \"Type\": \"Corresponding question type\"}\n"
                "##Reference Examples:\n"
                "{\"Q\": \"What if the people in the video doesn't put the food from the beginning into the pot and instead eat it directly, what would be the result?\", \"A\": \"The food shown at the beginning of the video is raw beef. Eating raw beef can pose significant health risks due to the potential presence of harmful bacteria such as E. coli, Salmonella, and Listeria. If the people in the video eat raw beef, they are likely to experience symptoms such as nausea, vomiting, diarrhea, abdominal pain, fever, and so on.\", \"Type\": \"Counterfactual Reasoning\"}\n"
                "{\"Q\": \"Which country does the food eaten by the man in the video come from?\", \"A\": \"The food the man in the video is eating is hot pot. Hot pot originated from China.\", \"Type\": \"Commonsense Reasoning\"}\n"
                "{\"Q\": \"Why does the boy stand on one leg after moving to the front?\", \"A\": \"This boy is striking a pose.\", \"Type\": \"Causal Reasoning\"}\n"
                "{\"Q\": \"Imagine if the animal on the man's hand in the video bites his arm, what would happen?\", \"A\": \"The animal in the video is a Burmese python. Burmese pythons are non-venomous snakes. Therefore, if the man in the video is bitten, he will feel pain but will not be poisoned.\", \"Type\": \"Counterfactual Reasoning\"}\n"
                "{\"Q\": \"What is the ride in the center of the screen at the beginning of the video?\", \"A\": \"The ride at the beginning of the video is the London Eye.\", \"Type\": \"Commonsense Reasoning\"}\n"
                "{\"Q\": \"How does the woman make a hole in the paper bag?\", \"A\": \"She uses scissors to make a hole in the paper bag.\", \"Type\": \"Causal Reasoning\"}\n"

        },
        {
            "role": "user",
            "content": contents
        }
    ]
    print(title_text)
    try:
        completion_0 = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
        )
        response_message_0 = completion_0.choices[0].message.content
        total_money_used = completion_0.usage.prompt_tokens * 5 / 1000000 + completion_0.usage.completion_tokens * 15 / 1000000
        print(response_message_0)

        try:
            result = response_message_0.replace("\n", "")
            result = "{" + re.findall(r'\{([^}]+)\}', result)[0] + "}"
            result_dict = literal_eval(result)
            task_type = result_dict["Type"]
            print(task_type)

            if task_type == "Counterfactual Reasoning":
                Counterfactual_Reasoning_Count += 1
            elif task_type == "Commonsense Reasoning":
                Commonsense_Reasoning_Count += 1
            elif task_type == "Causal Reasoning":
                Causal_Reasoning_Count += 1
            print(f"Counterfactual Reasoning {Counterfactual_Reasoning_Count}")
            print(f"Commonsense Reasoning {Commonsense_Reasoning_Count}")
            print(f"Causal Reasoning {Causal_Reasoning_Count}")
        except:
            pass

        print(completion_0.usage.completion_tokens)
        print(completion_0.usage.prompt_tokens)
        print(total_money_used)

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

# 当前进程数设置为1
Counterfactual_Reasoning_Count = 0
Commonsense_Reasoning_Count = 0
Causal_Reasoning_Count = 0
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = []

    file_list = list(os.listdir(base_dir))
    # print(len(file_list))
    file_list = sorted(file_list)
    # file_list = file_list[:100]

    total_dirs = len(file_list)
    global progress_bar
    progress_bar = tqdm(total=total_dirs, desc="Processing")

    for frame_dir_name in file_list:
        futures.append(executor.submit(gpt_forward, frame_dir_name))
