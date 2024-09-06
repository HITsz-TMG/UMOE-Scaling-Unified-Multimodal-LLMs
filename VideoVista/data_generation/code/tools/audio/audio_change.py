"""

"""

import os
import json
import math
import argparse

parser = argparse.ArgumentParser(description="Merge")
parser.add_argument("--audio_origin", type=str)
parser.add_argument("--audio_save", type=str)
args = parser.parse_args()

data = json.load(open(args.audio_origin, "r"))

results = {}
for idx, line in enumerate(data):
    video_id = line["video_id"]

    audio_transcript = ""
    times = []
    try:
        # 这里的segment就像是一个个的字幕，每个字幕都有一个开始时间和结束时间，并且还会登记 说话者的信息 speaker1 / speaker2
        for segment in line["segments"]:
            if "speaker" in segment:
                speaker = segment["speaker"]
            else:
                speaker = "none"

            text = segment["text"].strip()
            print(text)
            start_time = math.floor(segment['start'])
            end_time = math.floor(segment['end'])

            times.append([start_time, end_time])

            # 再次检查，防止有视频漏过40s的长度限制
            if end_time - start_time > 40:
                print(video_id)

            audio_transcript += f"Second {start_time} to {end_time}: {text.strip()}\n"
    except:
        audio_transcript = ""

    meta_info = {
        # "video_caption": video_caption,
        "audio_transcript": audio_transcript,
        "times": times
    }
    # print(meta_info)

    results[line["video_id"]] = meta_info

print(len(results))
json.dump(results, open(args.audio_save, "w"), indent=2)

