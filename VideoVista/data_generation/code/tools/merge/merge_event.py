import os
import json
import argparse

parser = argparse.ArgumentParser(description="Merge")
parser.add_argument("--timestamp", type=str)
parser.add_argument("--id2time", type=str)
parser.add_argument("--event", type=str)
parser.add_argument("--frames", type=str)
parser.add_argument("--save", type=str)
args = parser.parse_args()

merge_source = json.load(open(args.timestamp, "r"))
id2event = json.load(open(args.event, "r"))
id2time = json.load(open(args.id2time, "r"))
base_path = args.frames
print(len(merge_source))
results = []

count = 0
for line in merge_source:
    video_name = line["video_name"]
    times = line["time"]

    current_time = 0
    all_events = []
    for time in times:
        video_id = f"{video_name}.{time}"
        frame_path = os.path.join(base_path, video_id)
        # video_length = len(os.listdir(frame_path))
        video_length = id2time[video_id]
        try:
            events = id2event[video_id]
        except:
            current_time += video_length
            # print(video_id)
            assert len(os.listdir(frame_path)) == 0
            continue
            # events = [{"Event": ""}]

        event_text = ""

        if isinstance(events, list):
            for event in events:
                event_text += event["Event"] + " "
        else:
            assert "ResponsibleAIPolicyViolation" in events or "input image may contain content that is not allowed" in events
            count += 1
        event_text = event_text.strip()

        events_dict = {
            "Time": [round(current_time), round(current_time + video_length)],
            "Event": event_text
        }
        all_events.append(events_dict)
        current_time += video_length

    next_time = times[-1] + 1
    try:
        video_id = f"{video_name}.{next_time}"
        next_event = id2event[video_id]
        a = 1
    except:
        next_event = "none"

    # print(all_events)

    tmp = {
        "video_name": video_name,
        "times": times,
        "event": all_events,
        "next_event": next_event
    }
    results.append(tmp)

# print(count)
json.dump(results, open(args.save, "w"), indent=2)


