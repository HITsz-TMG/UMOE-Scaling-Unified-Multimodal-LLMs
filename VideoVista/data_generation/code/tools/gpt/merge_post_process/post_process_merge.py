import re
import os
import ast
import json
import argparse

parser = argparse.ArgumentParser(description="action recongnition")
parser.add_argument("--input", type=str)
parser.add_argument("--save", type=str)
parser.add_argument("--error_save", type=str, help="save error video id")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.save), exist_ok=True)
os.makedirs(os.path.dirname(args.error_save), exist_ok=True)

# --input processed/meta/merge/reannotate_actions/60t120

# --save processed/meta/QA_pairs/60t120/60t120_actions_QA.json

# --error_save processed/meta/error/merge_annotate/60t120/error_annotate_action_60t120.json

pattern = r'```[\s]*python(.*)```'
pattern_py = r'```[\s]*(.*)```'
pattern_json = r'```json(.*)```'
pattern_plaintext = r'plaintext\[(\{.*?\})\]'

error_videos = []
cnt = 0
results = []

path = args.input

for filename in os.listdir(path):
    video_id = filename.split(".")[0]
    time = filename.split(".")[1]
    idx = filename.split(".")[2]

    filepath = os.path.join(path, filename)
    data = json.load(open(filepath, "r"))

    video_name = data["video_name"]
    result = data["result"]
    origin_result = result


    result = result.replace("\n", "")
    result = result.replace("2. {", "")
    result = result.replace("3. {", "")

    if "'s" in result:
        result = result.replace("\\'s", "'s")
        result = result.replace("'s", "\\'s")


    if "python" in result and "```" in result:
        match = re.search(pattern, result)
        if match:
            result = match.group(1)

    if "json" in result and "```" in result:
        match = re.search(pattern_json, result)
        if match:
            result = match.group(1)

    if "```" in result:
        match = re.search(pattern_py, result)
        if match:
            result = match.group(1)

    if "plaintext" in result:
        match = re.search(pattern_plaintext, result)
        if match:
            result = "[" + match.group(1) + "]"

    after_result = result


    try:
        result_list = ast.literal_eval(result)
        # 到此处已经能锁定到一个具体的qa
        for result in result_list:
            question = result['Q']
            answer = result['A']
            QAtype = result['Type']
            timestamp = time
            QA_video = video_name

            tmp = {
                "origin_video_id": video_id,
                "video_id": video_name,
                "timestamp": time,
                "idx": idx,
                "question": question,
                "answer": answer,
                "QAtype": QAtype
            }

            results.append(tmp)

    except:
        print(result)
        print(video_name)
        error_videos.append(video_name)
        cnt += 1
        continue

json.dump(results, open(args.save, "w"), indent=2)
json.dump(error_videos, open(args.error_save, "w"), indent=2)












