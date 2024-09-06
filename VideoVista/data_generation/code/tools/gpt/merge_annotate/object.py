# 合并Objects形式的QA问题
# 主要通过GPT4进行实现

import os
from openai import OpenAI
import json
import time
import ast
import argparse
import warnings
from tqdm import tqdm

# Suppressing all warnings
# warnings.filterwarnings('ignore')
client = OpenAI(
    base_url="", api_key=""
)
model = "gpt-4-0125-preview"


parser = argparse.ArgumentParser(description="merge event")
parser.add_argument("--time", type=str)
parser.add_argument("--events", type=str)
parser.add_argument("--objects", type=str)
parser.add_argument("--save", type=str)
parser.add_argument("--error", type=str)
parser.add_argument("--video_titles", type=str)
parser.add_argument("--video_categorys", type=str)


args = parser.parse_args()

timestamp = args.time
data = json.load(open(args.objects, "r"))
events_data = json.load(open(args.events, "r"))

try:
    error_videos = json.load(open(args.error, "r"))
except:
    error_videos = []

id2titles = json.load(open(args.video_titles, "r"))
id2category = json.load(open(args.video_categorys, "r"))

base_save_path = args.save
if not os.path.exists(base_save_path):
    os.makedirs(base_save_path)

def annotate(videoid, clipid, objects_infos, events_infos):
    save_path = os.path.join(base_save_path, f"{clipid}.json")

    if os.path.exists(save_path) and clipid not in error_videos:
        return None

    title = id2titles[videoid]
    category = id2category[videoid][0]
    messages = [
        {
            "role": "system",
            "content":

            # 需要输入事件
            #
                "You're an AI visual assistant specialized in analyzing videos.\n"
                "You will receive a series of Question and Answer Pairs in the form of list of JSON dictionary and Event Information to help you understand the video.\n"
                # "# For the Question and Answer Pairs:\n"
                "# For Question and Answer Pairs:\n"
                "The original video is divided into several segments of varying lengths, and for each segment, we provide three questions along with their corresponding answers and time position of the video segment in the original video.\n"
                # "The information corresponding to each video segment consists of two parts: 1) The corresponding time position of the segment in the original video. 2) The three questions and answers based on video segment.\n"
                "All of input questions can be categorized into the following eight task types:\n"
                "1. Objects Existence. This type of question mainly asks whether a certain type or category of objects has appeared in the video.\n"
                "2. Objects Count. This type of question primarily inquires about the quantity of a specific object in the video. Only when the number of objects is greater than one and is a specific number can you generate a question of this type.\n"
                "3. Objects Temporal Location. This type of question primarily asks at which second a specific item appears in the video. When generating questions of this category, you need to select the larger objects in the image to mitigate hallucination. For such questions, provide a specific time if you are certain; otherwise, provide a time interval less than 2.\n"  # range, specific
                "4. Objects Spatial Location. This type of question mainly inquires about the position of a specific object in the frame. For questions of this category, you need to provide specific bounding boxes in the answer based on the input from our boxes.\n"
                "5. Objects Spatial Tracking. This type of question primarily asks how a specific object is moving. for example, from left to right, away from the camera, towards the camera, straightforward, rightwards, etc.\n"  # More
                "6. Objects Temporal Relation. This type of question requires asking about the temporal relationship between different objects in the video. For example, which object appears first, which one disappears first, etc.\n"
                "7. Objects Spatial Relation. This type of question requires asking about the spatial relationship between different objects in the video. For example, what position is object A in relative to object B?\n"
                "8. Optical Character Recognition. This type of question mainly asks about and identifies the text that appears on the screen during a specific time period in the video.\n"
                "# For Event Information:\n"
                "1. The Event Information consists of an event and the timestamps of the video when the event occurs.\n"
                # "2. The input event may contain hallucination. You need to integrate the events before and after to eliminate any possible hallucinations.\n"
                "2. The Event Information can help you understand the video, thereby mitigating the possible hallucinations in the question and answer pairs.\n"
                # "3. "
                "-----\n"
                "##Task Instruction:\n"
                "You need to merge and modify the input questions based to the following steps. \n"
                "Step 1. Modify all directly mentioned times in both questions and answers.\n"
                # "1) For questions or answers that directly mention time, you need to add the corresponding start time of the video segment to ensure the accuracy of the question and answer under the original video.\n"
                "1) For questions or answers that directly mention time, you need to add the **start time of the video segment** to the time mentioned in the question or answer, ensuring the accuracy of the question and answer under the original video.\n"
                "2) For 'start' and 'end' in questions and answers, you need to replace them with the start and end times of the corresponding video segment.\n"
                "Step 2. Determine whether questions of the same task type can be merged.\n"
                "1) For questions of Optical Character Recognition, you can merge them directly. And it is necessary to add time segments in the answers\n"  # And it is necessary to add time segments in the answers.  # 不同objcts不合并
                "2) For questions of Objects Existence, Objects Temporal Location, Objects Spatial Tracking and Objects Spatial Relation, if the task type of the questions and the queried objects are completely identical, you can merge them under the same type. For questions inquiring about different objects, you don't need to merge them.\n"
                "Step 3. Determine whether it is necessary to add a time range in the question.\n"
                # "1) When the question already contains specific time or time range or when the question is asking about time, skip Step 3 and don't add a time range in the question.\n"
                "1) When the question already contains specific time or when the question is asking about time, don't add a time range in the question.\n"
                "2) For questions of Objects Count, add a time range in the question to ensure the accuracy of the answer. For example, in the video from 11 to 16 seconds, how many elephants appeared in total?\n"
                "3) For questions of Objects Temporal Relation, add a time range in the question to ensure the accuracy of the answer. For example, within 10 to 39 seconds in the video, which one appeared first, the dog or the cat?\n"
                "4) For questions of Optical Character Recognition that have not been merged, if the question does not include time but the answer includes a time, you can move the time mentioned in the answer to the question.\n"
                "5) Except for the above three types, questions from the remaining types should not include a time range!!\n"  # "5) Do not add time ranges for the remaining questions unless necessary.\n"
                "-----\n"
                "##Output Instruction:\n"
                "Please generate the response in the form of a Python list of dictionary string with keys \"Q\" for question, \"A\" for answer and \"Type\" for question type.\n"
                "The time interval provided during input is no longer required during output; you only need to output the merged and modified question.\n"
                "When using the information from the event information, directly explain the scene, and do not mention that the information source is the event information. Always answer as if you are directly looking at the video.\n"
        },
        {
            "role": "user",
            "content":
                "Question and Answer Pairs:\n"
                f"{objects_infos}"
                "Event Information:\n"
                f"{events_infos}"
        }
    ]
    print(messages[0]["content"])
    try:
        completion_0 = client.chat.completions.create(
            model=model,
            messages=messages
        )

        response_message_0 = completion_0.choices[0].message.content
        total_money_used = completion_0.usage.prompt_tokens * 10 / 1000000 +  completion_0.usage.completion_tokens * 30 / 1000000
        print(completion_0.usage.completion_tokens)
        print(completion_0.usage.prompt_tokens)
        print(total_money_used)
        print(response_message_0)
        result = {
            "video_name": clipid,
            "result": response_message_0
        }

    except Exception as e:
        print(f"Video: {clipid}, Error: {e}")
        result = {
            "video_name": clipid,
            "result": f"Error: {e}"
        }

    json.dump(result, open(save_path, "w"))


def main():
    with tqdm(total=len(data)) as pbar:
        for idx, line in enumerate(data):
            video_id = line["video_name"]
            clip_id = f"{video_id}.{timestamp}.{idx}"

            events_infos = events_data[idx]["event"]
            objects_infos = line["objects"]

            if len(objects_infos) > 60:
                pbar.update(1)
                continue

            annotate(video_id, clip_id, objects_infos, events_infos)

            pbar.update(1)



if __name__ == "__main__":
    main()