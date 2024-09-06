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

# pattern_easy = r'\[(.*)\]'
error_videos = []
type2count = {}
path = args.input
results = {}

cnt = 0
for filename in os.listdir(path):
    filepath = os.path.join(path, filename)

    data = json.load(open(filepath, "r"))
    result = data["result"]


    # if "an unsupported image" in result or "Request timed out." in result or 'Connection error.' in result:
    #     continue

    result = result.replace("\n", "")

    try:
        result = "{" + re.findall(r'\{([^}]+)\}', result)[0] + "}"
        result_dict = ast.literal_eval(result)
    except:
        fileid = filename.replace("_qa_test.json", "")
        if "Your input image may contain content that is not allowed by our safety system." in result or "ResponsibleAIPolicyViolation" in result:
            result_dict = result
            print("error 400", fileid)
        else:
            print("other error", fileid)
            error_videos.append(fileid)
            continue

    video_name = data["video_name"]
    results[video_name] = result_dict

print(len(results))
json.dump(results, open(args.save, "w"), indent=2)
print(len(error_videos))
json.dump(error_videos, open(args.error_save, "w"), indent=2)