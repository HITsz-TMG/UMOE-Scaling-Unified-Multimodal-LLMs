# 对gpt4-o生成的结果进行后处理
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

pattern = r'```[\s]*python(.*)```'
pattern_py = r'```[\s]*py(.*)```'
pattern_json = r'```json(.*)```'

# pattern_easy = r'\[(.*)\]'

type2count = {}
path = args.input

error_videos = []
results = {}

cnt = 0
for filename in os.listdir(path):
    filepath = os.path.join(path, filename)

    # if "6oXnxw-ny2o.27" in filename:
    #     a = 1

    data = json.load(open(filepath, "r"))
    result = data["result"]
    # if "ResponsibleAIPolicyViolation" in result:
    #     continue

    result = result.replace("\n", "")
    # result = result.replace("\"", "'")
    result = result.replace("\'Event\': \'", "\"Event\": \"")
    # result = result.replace()


    result = re.sub(r"'\s*}", "\"}", result)

    if "python" in result and "```" in result:
        match = re.search(pattern, result)
        if match:
            result = match.group(1)

    if "json" in result and "```" in result:
        match = re.search(pattern_json, result)
        if match:
            result = match.group(1)

    if "py" in result and "```" in result:
        match = re.search(pattern_py, result)
        if match:
            result = match.group(1)


    try:
        result_list = ast.literal_eval(result)
    except:
        if "ResponsibleAIPolicyViolation" in result or "content_policy_violation" in result:
            result_list = result
        else:
            print(result)
            fileid = filename.replace(".json", "")
            print(fileid)
            error_videos.append(fileid)
            continue

    # tmp = {"video_name": data["video_name"],
    #     "event": result_list}
    # results.append(tmp)
    if "_events_s.json" in filename:
        video_name = filename.replace("_events_s.json", "")
    else:
        video_name = filename.replace(".json", "")
    results[video_name] = result_list

# print(results["6oXnxw-ny2o.27"])
print(len(results))
json.dump(results, open(args.save, "w"), indent=2)
print(len(error_videos))
json.dump(error_videos, open(args.error_save, "w"), indent=2)