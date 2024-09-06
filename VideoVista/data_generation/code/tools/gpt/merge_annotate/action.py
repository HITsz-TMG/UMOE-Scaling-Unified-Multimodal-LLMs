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


parser = argparse.ArgumentParser(description="merge action")
parser.add_argument("--time", type=str)
parser.add_argument("--actions", type=str)
parser.add_argument("--events", type=str)
parser.add_argument("--audios", type=str)
parser.add_argument("--save", type=str)
parser.add_argument("--error", type=str)
parser.add_argument("--video_titles", type=str)
parser.add_argument("--video_categorys", type=str)


args = parser.parse_args()

timestamp = args.time

data = json.load(open(args.actions, "r"))
events_data = json.load(open(args.events, "r"))
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


def annotate(videoid, clipid, action_infos, event_infos, audio_infos, future_infos):
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
                "You will receive Action Information and Event Information in the form of JSON dictionary. Additionally, we provide you with some additional information about the video: Audio Transcripts, Future Action Information, Title and Category.\n"
                "# For the Action Information:\n"
                "1. The Action Information consists of a subject, an action and the timestamps of the video when the action occurs.\n"
                "2. The input actions may contain errors. You need to consider the actions in the preceding and following time periods, to determine and remove error actions.\n"
                "3. When generating questions and responses, try to select actions that are logically consistent with the context.\n"
                "4. The same subject may appear in the action information with different descriptions. You need to merge the subjects in the action information."
                "# For the Event Information:\n"
                "1. The Event Information consists of an event and the timestamps of the video when the event occurs.\n"
                "2. The input event may contain hallucination. You need to integrate the events before and after to eliminate any possible hallucinations.\n"
                "3. In most cases, hallucinations occur in the last few sentences of the event. Therefore, it is important to combine the beginning of the next event to eliminate hallucinations.\n"
                "# For the Audio Transcripts:\n"
                "1. The audio transcript may be none, and its format is starting with 'Second {begin_second} to {end_second}'.\n"
                "2. The audio transcript may be noisy. When using it, you need to judge for yourself whether the information in it is valid.\n"
                "# For the Future Action:\n"
                "1. The future action contains the actions that will occur flollowing inputting video.\n"
                "2. This future action is only needed when you are constructing an 'Action Prediction' task; otherwise, ignore this part of the information.\n"
                "3. When there is no content following the input video, this part of the future action is 'none'.\n"
                "# For the Title:\n"
                "1. The video title provides insight into the video's main theme but shouldn't be solely relied upon for action information.\n"
                "2. Note that the video we input is a clip from the original video, while the video title we input is the full title of the original video.\n"
                "# For the Category:\n"
                "The video category is self-labeled by users upon upload, hence errors may exist.\n"
                "-----\n"
                "##Task Type:\n"
                "1. Action Recognition. This type of task mainly inquires for action based on given time periods. Note that the answer is preferably specific actions rather than scene description. Besides, you need to provide an approximate tim range in the question. "
                "Example: What action does the person in blue perform between the 2 and 11 second of the video?\n"
                "2. Action Location. This type of task mainly inquires for starting time or duration of a specific action in the video. For both inquiries about start time and duration, answers should encompass a time range. For questions asking about the starting time, the size of the time interval provided in the answer should not exceed 5. "
                "Example: At what time in the video does the man put on a gray shirt in front of the van?\n"
                "3. Action Sequence. This type of task requires providing a brief description of all actions occurring in the video and arranging them in order. In response, the detected actions must be labeled with numbers like 1), 2), 3). If the actions before and after are the same, they need to be merged. "
                "For videos longer than ten minutes, then action sequences you provide should be around 10. If there are too many actions in the video, you only need to provide the main actions.\n"
                "Example: In what sequence do the actions performed in the video?\n"
                "4. Action Prediction. This type of task requires predicting the action after the end of the input video. You can refer to 'Future Action' for anticipating the answer. If the future action is 'none', you should predict it based on events informations. "
                "Example: Please anticipate what action will occur following the end of this video.\n"
                "5. Action Count. This type of question mainly ask about the number of actions required to **complete a task** in the video. It should be noted that when answering such questions, you need to merge similar actions and only retain actions with a certain degree of distinction. "
                "Example: How many different actions does the man in the video perform to slice beef?\n"
                "-----\n"
                "##Task Instruction:\n"
                "You need to create a question and its corresponding answer for each of the five Task Types mentioned above.\n"
                "Ensure that the questions can be answered from the action and event information.\n"
                "When generating questions and answering questions, event information can help you better understand the entire video.\n"
                "When using the information from the action information, directly explain the scene, and do not mention that the information source is the action information. Always answer as if you are directly looking at the video.\n"
                "When referring to multiple characters in the question, it is necessary to add certain adjectives, such as clothing, etc.\n"
                "In generated answer, you can use 'second' to indicate time, but don't mention 'frame'.\n"
                "Please generate the response in the form of a Python list of dictionary string with keys \"Q\" for question and \"A\" for answer. Each corresponding value should be the question and answer text respectively.\n"
                "For example, your response should look like this: [{\"Q\": \"Your first question here...\", \"A\": \"Your first answer here...\", \"Type\": \"Corresponding question type\"}, ..., {\"Q\": \"Your third question here...\", \"A\": \"Your third answer here...\", \"Type\": \"Corresponding question type\"}]\n"
                "[{\"Q\": \"What action does the person with a ring perform between the first and fourth second of the video?\", \"A\": \"The person with a ring on their left hand is whisking eggs in a white bowl with a wire whisk.\", \"Type\": \"Action Recognition\"}, "
                "{\"Q\": \"When does the person start pouring the egg mixture into the pan?\", \"A\": \"The subject starts pouring the egg mixture into the pan from 19 seconds to 24 seconds.\", \"Type\": \"Action Location\"}, "
                "{\"Q\": \"Can you describe the sequence of actions performed from the beginning to the end of the video?\", \"A\": \"1) A person with a ring on their left hand starts whisking eggs in a white bowl with a wire whisk. 2) The same person continues to whisk the eggs, making sure they are all broken up. 3) The person then prepares a pan with butter and bacon, and adds the egg mixture. 4) Using a spatula, the person stirs and cooks the scrambled eggs in the pan, continuously ensuring even cooking. 5) Finally, scrambled eggs are transferred out of the frying pan.\", \"Type\": \"Event Location\"}, "
                "{\"Q\": \"What action will happen right after this video comes to an end?\", \"A\": \"none\", \"Type\": \"Action Prediction\"}, "
                "{\"Q\": \"How many actions does it take for the person in the video to fry an egg?\", \"A\": \"The person in the video requires four different actions to fry the egg:  whisking the eggs in a glass bowl, pouring the eggs from the bowl into a frying pan, flipping and stirring the eggs in the frying pan, and plating the cooked eggs.\", \"Type\": \"Action Count\"}]"
        },
        {
            "role": "user",
            "content":
                "Action Information:\n"
                f"{action_infos}\n"
                "Event Information:\n"
                f"{event_infos}\n"
                "Audio Transcripts:\n"
                f"{audio_infos}\n"
                "Future Action:\n"
                f"{future_infos}\n"
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
    id2actions = {}
    for line in data:
        id2actions[f"{line['video_name']}.{line['times']}"] = line

    with tqdm(total=len(data)) as pbar:
        for idx, line in enumerate(events_data):

            video_id = line["video_name"]
            mapping_id = f"{line['video_name']}.{line['times']}"

            clip_id = f"{video_id}.{timestamp}.{idx}"

            events_infos = line["event"]
            audio_infos = audio_data[idx]["audio"]

            # mapping_id 是基于event基础上的 // 而id2actions是基于action的
            if mapping_id not in id2actions:
                continue

            action_infos = id2actions[mapping_id]["action"]

            future_infos = id2actions[mapping_id]["next_action"]
            annotate(video_id, clip_id, action_infos, events_infos, audio_infos, future_infos)

            pbar.update(1)


if __name__ == "__main__":
    main()



##