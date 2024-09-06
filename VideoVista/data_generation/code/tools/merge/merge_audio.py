import os
import json
import math
import argparse

parser = argparse.ArgumentParser(description="Merge")
parser.add_argument("--timestamp", type=str)
parser.add_argument("--id2time", type=str)
parser.add_argument("--audio", type=str)
parser.add_argument("--frames", type=str)
parser.add_argument("--save", type=str)
args = parser.parse_args()

merge_source = json.load(open(args.timestamp, "r"))
id2time = json.load(open(args.id2time, "r"))
base_path = args.frames
audios = json.load(open(args.audio, "r"))

id2audio = {}
for audio in audios:
    id2audio[audio["video_id"]] = audio

print(len(merge_source))
results = []
for line in merge_source:
    video_name = line["video_name"]
    times = line["time"]

    current_time = 0
    audio_transcript = ""
    for time in times:
        video_id = f"{video_name}.{time}"
        frame_path = os.path.join(base_path, video_id)
        video_length = id2time[video_id]
        # try:
        current_audios = id2audio[video_id]
        if "segments" in current_audios:
            for segment in current_audios["segments"]:
                text = segment["text"].strip()
                start_time = round(current_time + math.floor(segment['start']))
                end_time = round(current_time + math.floor(segment['end']))
                audio_transcript += f"Second {start_time} to {end_time}: {text.strip()}\n"

        current_time += video_length

    tmp = {
        "video_name": video_name,
        "times": times,
        "audio": audio_transcript
    }
    results.append(tmp)

print(len(results))
json.dump(results, open(args.save,"w"), indent=2)


