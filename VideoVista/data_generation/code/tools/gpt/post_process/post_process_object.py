# 对gpt4o生成的结果进行后处理
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

path = args.input

error_videos = []
type2count = {}
results = {}
cnt = 0
for filename in os.listdir(path):

    filepath = os.path.join(path, filename)
    data = json.load(open(filepath, "r"))

    # if "qa_test" not in filename:
    #     video_name = filename.replace(".json", "")
    #     results[video_name] = data
    #     continue


    result = data["result"]
    # if "Error" in result or "Request timed out." in result or 'Connection error.' in result:
    #     continue

    result = result.replace("\n", "")
    result = result.replace("2. {", "")
    result = result.replace("3. {", "")

    if "'s" in result:
        result = result.replace("\\'s", "'s")
        result = result.replace("'s", "\\'s")

    # match = re.search(pattern_easy, result)
    # if match:
    #     result = match.group(1)
    #     result = f"[{result}]"

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
        for r in result_list:
            if r["Type"] == "Objects Counts":
                r["Type"] = "Objects Count"

    except:
        print(result)
        fileid = filename.replace("_qa_test.json", "")
        # print(fileid)
        if "Your input image may contain content that is not allowed by our safety system." in result or "ResponsibleAIPolicyViolation" in result:
            print("get in post", result)
            result_list = result
        elif result.lower() == "none":
            result_list = "none"
        else:
            error_videos.append(fileid)
            print(fileid)
            print(result)
            cnt += 1
            continue

    video_name = data["video_name"]
    results[video_name] = result_list
    # tmp = {"video_name": data["video_name"],
    #        "questions": result_list}
    # results.append(tmp)
    # for r in result_list:
    #     type = r["Type"]
    #     if type not in type2count:
    #         type2count[type] = 0
    #     type2count[type] += 1

    #     tmp = {"video_name": data["video_name"],
    #            "Q": r["Q"],
    #            "A": r["A"],
    #            "Type": type}

    #     results.append(tmp)

# print(cnt)
# print(len(results))
# print(type2count)
print(len(results))
json.dump(results, open(args.save, "w"), indent=2)
print(len(error_videos))
json.dump(error_videos, open(args.error_save, "w"), indent=2)