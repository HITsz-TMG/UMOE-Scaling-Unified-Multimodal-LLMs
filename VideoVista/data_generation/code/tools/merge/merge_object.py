import os
import json
import argparse

parser = argparse.ArgumentParser(description="Merge")
parser.add_argument("--timestamp", type=str)
parser.add_argument("--id2time", type=str)
parser.add_argument("--object", type=str)
parser.add_argument("--frames", type=str)
parser.add_argument("--save", type=str)
args = parser.parse_args()

merge_source = json.load(open(args.timestamp, "r"))
id2object = json.load(open(args.object, "r"))
id2time = json.load(open(args.id2time, "r"))
base_path = args.frames

print(len(merge_source))
results = []
for line in merge_source:
    video_name = line["video_name"]
    times = line["time"]

    # video_name : videoid
    # times : [0,1,2,3,4,5]
    current_time = 0
    all_objects = []
    for time in times:
        video_id = f"{video_name}.{time}"
        frame_path = os.path.join(base_path, video_id)
        # video_length = len(os.listdir(frame_path))
        video_length = id2time[video_id]
        try:
            objects = id2object[video_id]
        except:
            assert len(os.listdir(frame_path)) == 0
            current_time += video_length
            continue

        if not isinstance(objects, list):
            assert "input image may contain content that is not allowed" in objects or "ResponsibleAIPolicyViolation" in objects or "none" in objects
        elif "Time" in objects[0]:
            for line in objects:
                line["Time"][0] += round(current_time)
                line["Time"][1] += round(current_time)
            all_objects += objects
        else:
            # event_text = event_text.strip()
            objects_dict = {
                "Time": [round(current_time), round(current_time + video_length)],
                "QAs": objects
            }
            all_objects.append(objects_dict)

        current_time += video_length

    print(all_objects)

    tmp = {
        "video_name": video_name,
        "times": times,
        "objects": all_objects
    }
    results.append(tmp)

# print(count)
json.dump(results, open(args.save,"w"), indent=2)





