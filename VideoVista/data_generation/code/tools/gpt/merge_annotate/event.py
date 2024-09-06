# 关于动作的QA Prompt
# 使用纯文本作为输入

# Required Libraries
import os
from openai import OpenAI
import json
import time
import ast
import argparse
import warnings
from tqdm import tqdm

# Suppressing all warnings
warnings.filterwarnings('ignore')
client = OpenAI(
    base_url="", api_key=""
)
model = "gpt-4-0125-preview"

parser = argparse.ArgumentParser(description="merge event")
parser.add_argument("--time", type=str)
parser.add_argument("--events", type=str)
parser.add_argument("--audios", type=str)
parser.add_argument("--save", type=str)
parser.add_argument("--error", type=str)
parser.add_argument("--video_titles", type=str)
parser.add_argument("--video_categorys", type=str)

args = parser.parse_args()

timestamp = args.time
data = json.load(open(args.events, "r"))
audio_data = json.load(open(args.audios, "r"))

try:
    error_videos = json.load(open(args.error, "r"))
except:
    error_videos = []



id2titles = json.load(open(args.video_titles, "r"))
id2category = json.load(open(args.video_categorys, "r"))

base_save_path = args.save
if not os.path.exists(base_save_path):
    os.makedirs(base_save_path)
def annotate(videoid, clipid, event_infos, audio_infos):
    save_path = os.path.join(base_save_path, f"{clipid}.json")

    if os.path.exists(save_path) and clipid not in error_videos:
        return None

    title = id2titles[videoid]
    category = id2category[videoid][0]
    messages = [
        {
            "role": "system",
            "content":
                "You're an AI visual assistant specialized in analyzing videos.\n"
                "You will receive event information in the form of JSON dictionary. Additionally, we provide you with some additional information about the video: Title, Category and Audio Transcripts.\n"
                "# For the event information:\n"
                "1. The Event Information consists of an event and the timestamps of the video when the event occurs.\n"
                "2. The input event may contain hallucination. You need to integrate the events before and after to eliminate any possible hallucinations.\n"
                "3. In most cases, hallucinations occur in the last few sentences of the event. Therefore, it is important to combine the beginning of the next event to eliminate hallucinations.\n"
                "#For the Audio Transcripts:\n"
                "1. The audio transcript may be none, and its format is starting with 'Second {begin_second} to {end_second}'.\n"
                "2. The audio transcript may be noisy. When using it, you need to judge for yourself whether the information in it is valid.\n"
                "# For the Title:\n"
                "1. The video title provides insight into the video's main theme but shouldn't be solely relied upon for action information.\n"
                "2. Note that the video we input is a clip from the original video, while the video title we input is the full title of the original video.\n"
                "# For the Category:\n"
                "The video category is self-labeled by users upon upload, hence errors may exist.\n"
                "-----\n"
                "##Task Type:\n"
                "1. Brief Description. This type of task requires providing a brief description of the video, with the description limited to one to two sentences.\n"
                "2. Detailed Description. This type of task requires providing a detailed description of the video, with the description limited to five to six sentences.\n"
                "3. Event Location. This type of task mainly inquires for starting time or duration of a specific event in the video. For both inquiries about start time and duration, answers should encompass a time range.\n"
                "4. Event Description. This type of task mainly asks how a specific event progresses in the video. This type of question usually involves only a single event in the video.\n"
                "5. Event Sequences. This type of task requires providing a brief description of all events occurring in the video and arranging them in order. In response, the detected events must be labeled with numbers like 1, 2, 3.\n"
                "-----\n"
                "##Task Instruction:\n"
                "You need to create a question and its corresponding answer for each of the five Task Types mentioned above.\n"
                "Ensure that the questions can be answered from the event information.\n"
                "When using the information from the event information, directly explain the scene, and do not mention that the information source is the event information. Always answer as if you are directly looking at the video.\n"
                "When referring to multiple characters in the question, it is necessary to add certain adjectives, such as clothing, etc.\n"
                "In generated answer, you can use 'second' to indicate time, but don't mention 'frame'.\n"
                "Please generate the response in the form of a Python list of dictionary string with keys \"Q\" for question and \"A\" for answer. Each corresponding value should be the question and answer text respectively.\n"
                "For example, your response should look like this: [{\"Q\": \"Your first question here...\", \"A\": \"Your first answer here...\", \"Type\": \"Corresponding question type\"}, ..., {\"Q\": \"Your third question here...\", \"A\": \"Your third answer here...\", \"Type\": \"Corresponding question type\"}]\n"
                "------\n"
                "##Reference Examples:\n"
                "[{\"Q\": \"What is the primary focus of the video?\", \"A\": \"The video primarily focuses on demonstrating how to cook great scrambled eggs, starting with whisking eggs and ending with scrambling them in a pan until they are ready to be served.\", \"Type\": \"Brief Description\"}, "
                "{\"Q\": \"Could you describe the process of making scrambled eggs as shown in the video?\", \"A\": \"The process begins with a person whisking eggs in a bowl to create a smooth, uniform mixture. Next, the mixture is poured into a pan prepped with butter (and a bit of bacon as mentioned), where it is continuously stirred with a spatula by a person in a patterned top. The stirring and folding technique ensures even cooking and prevents the eggs from sticking, resulting in light and fluffy scrambled eggs. The cooked eggs are finally transferred from the pan onto a plate for serving.\", \"Type\": \"Detailed Description\"}, "
                "{\"Q\": \"At approximately which second in the video does the person transfer the eggs to the plate?\", \"A\": \"At around 86-90 seconds into the video, the person transfers the eggs to the plate.\", \"Type\": \"Event Location\"}, "
                "{\"Q\": \"How does the person initially prepare the scrambled eggs before cooking them in the pan?\", \"A\": \"The person starts by whisking eggs in a bowl with a wire whisk, ensuring they are all broken up before pouring the egg mixture into the pan.\", \"Type\": \"Event Description\"}, "
                "{\"Q\": \"In what sequence do the events occur in the video?\", \"A\": \"1) Whisking eggs in a bowl using a wire whisk. 2) Adding butter to the pan and heating it. 3) Pouring the whisked eggs into the hot pan. 4) Using a spatula to repeatedly stir-fry the eggs in the pan. 5) Plate the cooked eggs and present them.\", \"Type\": \"Event sequences\"}]"
        },
        {
            "role": "user",
            "content":
            # "Action Information:\n"
            # f"{action_infos}\n"
                "Event Information:\n"
                f"{event_infos}\n"
                "Audio Transcripts:\n"
                f"{audio_infos}\n"
                "Title:\n"
                f"{title}\n"
                "Category:\n"
                f"{category}\n"
        }
    ]
    try:
        completion_0 = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048,
        )

        response_message_0 = completion_0.choices[0].message.content
        total_money_used = completion_0.usage.prompt_tokens * 10 / 1000000 + completion_0.usage.completion_tokens * 30 / 1000000
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
            "result": str(e)
        }

    json.dump(result, open(save_path, "w"))


def main():
    with tqdm(total=len(data)) as pbar:
        for idx, line in enumerate(data):
            video_id = line["video_name"]
            clip_id = f"{video_id}.{timestamp}.{idx}"

            # action_infos = line["action"]
            events_infos = line["event"]
            audio_infos = audio_data[idx]["audio"]
            annotate(video_id, clip_id, events_infos, audio_infos)

            pbar.update(1)


if __name__ == "__main__":
    main()

