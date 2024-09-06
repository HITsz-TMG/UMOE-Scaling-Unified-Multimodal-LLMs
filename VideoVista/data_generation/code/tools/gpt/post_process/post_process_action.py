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
pattern_py = r'```[\s]*(.*)```'
pattern_json = r'```json(.*)```'
pattern_plaintext = r'plaintext\[(\{.*?\})\]'
pattern_plaintext2 = r'```plaintext(.*)```'


type2count = {}
path = args.input

error_videos = []
cnt = 0
results = {}
for filename in os.listdir(path):

    filepath = os.path.join(path, filename)
    data = json.load(open(filepath, "r"))

    result = data["result"]

    result = result.replace("actions = ", "")

    result = result.replace("\n", "")

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


    try:
        result_list = ast.literal_eval(result)
    except:
        result = result.replace("\"", "'")
        result = result.replace("'Time'", "\"Time\"")
        result = result.replace("'Subject': '", "\"Subject\": \"")
        result = result.replace("', 'Action': '", "\", \"Action\": \"")
        result = result.replace("'}", "\"}")

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
                cnt += 1
                continue

    # assert isinstance(result_list[0], dict)
    # tmp = {"video_name": data["video_name"],
    #        "action": result_list}
    # results.append(tmp)

    video_name = filename.replace(".json", "")
    results[video_name] = result_list

# print(cnt)
print(len(results))
json.dump(results, open(args.save, "w"), indent=2)
print(len(error_videos))
json.dump(error_videos, open(args.error_save, "w"), indent=2)

