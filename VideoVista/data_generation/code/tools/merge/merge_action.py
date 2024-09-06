import os
import json
import argparse

parser = argparse.ArgumentParser(description="Merge")
parser.add_argument("--timestamp", type=str)
parser.add_argument("--id2time", type=str)
parser.add_argument("--action", type=str)
parser.add_argument("--frames", type=str)
parser.add_argument("--save", type=str)
args = parser.parse_args()

merge_source = json.load(open(args.timestamp, "r"))
id2action = json.load(open(args.action, "r"))
id2time = json.load(open(args.id2time, "r"))
base_path = args.frames

print(len(merge_source))
results = []
for line in merge_source:

    video_name = line["video_name"]
    times = line["time"]

    current_time = 0
    all_actions = []
    for time in times:
        video_id = f"{video_name}.{time}"
        frame_path = os.path.join(base_path, video_id)
        # video_length = len(os.listdir(frame_path))
        video_length = id2time[video_id]

        if video_length == None:
            video_length = 0

        try:
            actions = id2action[video_id]
        except:
            actions = [{"Time": [0, round(video_length)], "Subject": "none", "Action": "none"}]
            assert len(os.listdir(frame_path)) == 0
            # video_length = 1

        if isinstance(actions, list):
            for action in actions:
                action["Time"][0] += round(current_time)
                try:
                    action["Time"][1] += round(current_time)
                except:
                    a = 1
                all_actions.append(action)

        else:
            assert "ResponsibleAIPolicyViolation" in actions or "input image may contain content that is not allowed" in actions
            # all_actions.append(actions)
            a = 1
            pass

        current_time += video_length
    next_time = times[-1] + 1
    try:
        video_id = f"{video_name}.{next_time}"
        next_action = id2action[video_id]
        a = 1
    except:
        next_action = [{"Time": [0, 0], "Subject": "none", "Action": "none"}]

    # print(all_actions)

    tmp = {
        "video_name": video_name,
        "times": times,
        "action": all_actions,
        "next_action": next_action
    }
    results.append(tmp)

print(len(results))
json.dump(results, open(args.save,"w"), indent=2)


