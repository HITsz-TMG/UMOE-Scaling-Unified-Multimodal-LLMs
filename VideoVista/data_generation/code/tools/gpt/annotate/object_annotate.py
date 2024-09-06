import os
import argparse
from openai import OpenAI
import warnings
from PIL import Image, ImageDraw
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import base64
import json
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


# results = []


parser = argparse.ArgumentParser(description="object recognition")
parser.add_argument("--audios", type=str)
parser.add_argument("--frames", type=str)
parser.add_argument("--save", type=str)
parser.add_argument("--boxes", type=str)
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


def gpt_forward(frame_dir_name):
    frame_dir = os.path.join(base_dir, frame_dir_name)
    save_path = os.path.join(base_save_path, f"{frame_dir_name}")

    print("save_path", save_path)

    if os.path.exists(save_path) and frame_dir_name not in error_videos:
        return None

    video_length = len(os.listdir(frame_dir))
    print(video_length)
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
        audio_text = 'none\n'

    # Title
    title_text = video_titles[frame_dir_name.split(".")[0]]
    # category
    category_text = video_categorys[frame_dir_name.split(".")[0]][0]

    # boxes
    boxes_text = ""
    for second in range(0, len(images_list)):
        box_id = f"{frame_dir_name}.{second}"
        try:
            boxes = video_bounding_boxes[box_id]
            boxes_text += f"Second {second}: "
            for box in boxes:
                boxes_text += f"{box['label']}: {box['box']}"
                if box != boxes[-1]:
                    boxes_text += ", "

            boxes_text = boxes_text + "\n"
        except:
            boxes_text = "none\n"
    boxes_text = boxes_text.strip()
    contents.append(
        {
            "type": "text",
            "text": f"The input video contains a total of {len(images_list) - 1} seconds, so the number of frames in the input is {len(images_list)}.\n"
                    f"Video Title: {title_text}\n"
                    f"Video Category: {category_text}\n"
                    f"Boxes: {boxes_text}\n"
        }
    )

    messages = [
        {
            "role": "system",
            "content":
                "You are an AI visual assistant specialized in analyzing videos.\n"
                "You will receive a series of composed images and each composite image contains temporal frames. All composed images form a clip of a video. Additionally, we provide you with some additional information about the video: Title, Category, and Boxes.\n"
                "# For the video frames:\n"
                f"1. Except for the last image, each of the preceding images contains {frame_each_image} temporal frames from the video from left to right. The number of temporal frames contained in the last image is less than or equal to {frame_each_image}.\n"
                f"2. The total seconds are equal to the total number of frames. The first frame of first image corresponds to the 0 second of the video, the second frame of first image corresponds to the 1 second of the video, while the first frame of ith image corresponds to the i*{number_second} second of the video, and so on.\n"
                "# For the Title:\n"
                "The video title provides insight into the video's main theme but shouldn't be solely relied upon for action information."
                "# For the Category:\n"
                "The video category is self-labeled by users upon upload, hence errors may exist.\n"
                "#For the Boxes:\n"
                "1. Only use boxes information when you are generating Object Spatial Location questions!!!\n"
                "2. For the video frame at the {second} second of input, we provide corresponding the boxes, starting with 'Second {second}'.\n"
                "3. The boxes for each second is provided in the form of '{label}: {box}', and if there are multiple boxes, they are separated by ','.\n"
                "4. The label within the boxes may be error; you need to judge based on the image when using it."
                "-----\n"
                "##Task Types:\n"
                "1. Object Existence. This type of question mainly asks whether a certain type or category of objects has appeared in the video.\n"
                "2. Object Count. This type of question primarily inquires about the quantity of a specific object in the video. Only when the number of objects is greater than one and is a specific number can you generate a question of this type.\n"
                "3. Object Temporal Location. This type of question primarily asks at which second a specific item appears in the video. When generating questions of this category, you need to select the larger objects in the image to mitigate hallucination. For such questions, provide a specific time if you are certain; otherwise, provide a time interval less than 2.\n"  # range, specific
                "4. Object Spatial Location. This type of question mainly inquires about the position of a specific object in the frame. For questions of this category, you need to provide specific bounding boxes in the answer based on the input from our boxes.\n"
                "5. Object Spatial Tracking. This type of question primarily asks how a specific object is moving. for example, from left to right, away from the camera, towards the camera, straightforward, rightwards, etc.\n"  # More
                "6. Object Temporal Relation. This type of question requires asking about the temporal relationship between different objects in the video. For example, which object appears first, which one disappears first, etc.\n"
                "7. Object Spatial Relation. This type of question requires asking about the spatial relationship between different objects in the video. For example, what position is object A in relative to object B?\n"
                "8. Optical Character Recognition. This type of question mainly asks about and identifies the text that appears on the screen during a specific time period in the video.\n"
                "##Task Instruction:\n"
                "Your task is to use the provided information, create three questions about the video, and provide corresponding answers.\n"
                "When generating questions, you need to select three types of questions from the above task types that can provide accurate answers, and generate questions corresponding to these types.\n"
                "It's   important to note that both the questions and answers you provide need to be as accurate as possible.\n"
                "f you cannot create questions from the input visual content, then output 'none'\n"
                "When using the information from the frames or boxes, directly explain the scene, and do not mention that the information source is the frames or boxes. Always answer as if you are directly looking at the video.\n"
                "When referring to characters in the question, it is necessary to add certain adjectives, such as clothing, etc.\n"
                "In all the input information, the images formed by frames are the decisive information, while all other information serves as auxiliary information to help you understand the video.\n"
                "Please generate the response in the form of a Python list of dictionary string with keys \"Q\" for question, \"A\" for answer and \"Type\" for question type. Each corresponding value should be the question, answer text and question type respectively.\n"
                "For example, your response should look like this: [{\"Q\": \"Your first question here...\", \"A\": \"Your first answer here\", \"Type\": \"Corresponding question type\"}, {\"Q\": \"Your first question here...\", \"A\": \"Your first answer here\", \"Type\": \"Corresponding question type\"}, ...]\n"
                "##Reference Examples:\n"
                "[{\"Q\": \"Does cat appear in this video?\", \"A\": \"Yes, a cat appears in the second half of the video.\", \"Type\": \"Object Existence\"}, {\"Q\": \"At what second does the brown dog appear in the video?\", \"A\": \"The brown dog first appears in the video around 14-15 seconds.\", \"Type\": \"Object Temporal Location\"}, {\"Q\": \"Who appears first in the video, the dog or the cat?\", \"A\": \"The dog appears in the video before the cat.\", \"Type\": \"Object Temporal Relation\"}]\n"
                "[{\"Q\": \"At what second does the red ship appear in the video?\", \"A\": \"Around the 8th second of the video, we can see a red ship on the river.\", \"Type\": \"Object Temporal Locationh\"}, {\"Q\": \"When the red ship appears, where is it positioned in the video frame?\", \"A\": \"[0.62, 0.41, 1.0, 0.94].\", \"Type\": \"Object Spatial Location\"}, {\"Q\": \"In which direction is the ship driving?\", \"A\": \"The red ship is moving from right to left.\", \"Type\": \"Object Spatial Tracking\"}]\n"
                "[{\"Q\": \"Does a zookeeper appear in the video?\", \"A\": \"No, there is no zookeeper appearing in the video.\", \"Type\": \"Object Existence\"}, {\"Q\": \"How many elephants appear in total in the video?, \"A\": \"In total, two elephants appear in the video.\", \"Type\": \"Object Count\"}, {\"Q\": \"What animal is on the left side of the elephants?\", \"A\": \"A group of giraffes is positioned to the left of the elephants.\", \"Type\": \"Object Spatial Relation\"}]\n"
                "[{\"Q\": \"How many different items does the person show in the video?\", \"A\": \"The person shows two different items in the video: a small white bag and a brown beanie.\", \"Type\": \"Object Count\"}, {\"Q\": \"What objects are to the right front of the person in the video?\", \"A\": \"To the right front of the person is a MacBook laptop.\", \"Type\": \"Object Spatial Location\"}, {\"Q\": \"At what second does the person start showing the brown beanie?\", \"A\": \"The person starts showing the brown beanie around the 25th second.\", \"Type\": \"Object Temporal Location\"}]\n"
                "[{\"Q\": \"Who disappears from the video frame first, the person or the horse?\", \"A\": \"The person disappears first in the video.\", \"Type\": \"Object Temporal Relation\"}, {\"Q\": \"In what direction is the horse moving in the video?\", \"A\": \"The horse first moves to the left, and after a while, it turns around and begins to move to the right.\", \"Type\": \"Object Spatial Tracking\"}, {\"Q\": \"Does a horse appear in this video?\", \"A\": \"Yes, a horse appears in the video.\", \"Type\": \"Object Existence\"}]\n"
                "[{\"Q\": \"What is written on the sign above the entrance of the fortress?\", \"A\": \"The sign above the entrance of the fortress reads 'Medieval Times'.\", \"Type\": \"Optical Character Recognition\"}, {\"Q\": \"Does a person appear at the entrance of the fortress in the video?\", \"A\": \"Yes, multiple people appear at the entrance of the fortress in the video.\", \"Type\": \"Object Existence\"}, {\"Q\": \"At the first second of the video, what are the bounding box coordinates of the building?\", \"A\": \"The bounding box for the building is [0.00, 0.12, 0.98, 0.98]\", \"Type\": \"Object Existence\"}]\n"
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
        )
        response_message_0 = completion_0.choices[0].message.content
        total_money_used = completion_0.usage.prompt_tokens * 5 / 1000000 + completion_0.usage.completion_tokens * 15 / 1000000
        print(completion_0.usage.completion_tokens)
        print(completion_0.usage.prompt_tokens)
        print(total_money_used)
        print(response_message_0)
        result = {
            "video_name": frame_dir_name,
            "result": response_message_0
        }
        json.dump(result, open(save_path, "w"))

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
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = []

    file_list = list(os.listdir(base_dir))
    file_list = sorted(file_list)
    print("filelist", file_list)

    total_dirs = len(file_list)

    global progress_bar
    progress_bar = tqdm(total=total_dirs, desc="Processing")

    for frame_dir_name in file_list:
        futures.append(executor.submit(gpt_forward, frame_dir_name))
