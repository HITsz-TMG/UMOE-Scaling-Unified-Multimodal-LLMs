# 以一个动作丰富的clip向前或者向后扩展，直到所需要的长度

import os
import json
import argparse

parser = argparse.ArgumentParser(description="Merge")
parser.add_argument("--id2time_path", type=str)
parser.add_argument("--input_videos_path", type=str)
parser.add_argument("--save_path", type=str)
parser.add_argument("--save_name", type=str)

args = parser.parse_args()

id2time = args.id2time_path
id2time = json.load(open(id2time, "r"))
path = args.input_videos_path

count = 0

results_60t120 = []
results_120t300 = []
results_300t600 = []


results_600t900 = []
results_900t1200 = []
results_1200t1500 = []
results_1500t1800 = []
results_1800tu = []
results_full = []




for filename in os.listdir(path):

    print(filename)
    filepath = os.path.join(path, filename)
    num_clip = len(os.listdir(filepath))

    clip_list = list(os.listdir(filepath))
    clip_list = [(int(c.split(".")[1]), c) for c in clip_list]

    clip_list = sorted(clip_list, key=lambda x: x[0])

    value_list = [0] * (num_clip)
    end_list = [0] * (num_clip + 1)
    is_full = True

    for idx, clip_info in enumerate(clip_list):
        clip_name = clip_info[1]
        clip_id = clip_name.split(".")[0] + "." + clip_name.split(".")[1]
        # try:
        #     action = id2action[clip_id]
        #     cnt = 0
        #     for a in action:
        #         if a["Action"] != "none":
        #             cnt += 1
        #     # 简单判断这个clip是否有动作
        #     if cnt / len(action) > 0.9: # 丰富度
        #         count += 1
        #         value_list[idx] = 1

        # except:
        #     action = [{'Time': [0, 0], 'Subject': 'none', 'Action': 'none'}]

        # clip_frame_path = os.path.join(clip_frame, clip_id)
        # clip_time_length = len(os.listdir(clip_frame_path))
        # clip_time_length = max(1, clip_time_length) # 存在部分小于一秒的clip，我们将其设置为1秒

        clip_time_length = round(id2time[clip_id])
        end_list[idx + 1] = end_list[idx] + clip_time_length

    # 保证同一个长度下的视频没有交集，但是不同长度下的视频可以有交集

    # 一分钟到两分钟
    begin = 0
    interval = 8
    while begin < len(end_list) - 1:
        current_list = end_list[begin: begin + interval]
        # current_value_list = value_list[begin: begin + interval - 1]
        l = current_list[-1] - current_list[0]
        print(l,current_list)

        # value_score = sum(current_value_list) / len(current_value_list)

        if l <= 60:
            begin += 1
            interval = 8
        elif l >= 120:
            interval -= 1
            if interval < 3:
                begin += 1
                interval = 8
        else:
            if list(range(begin, begin + interval - 1)) == list(range(0, len(end_list)-1)):
                    is_full = True
            else:
                    is_full = False
            # 符合条件
            print("此时的interval", interval)
            print("begin", begin)
            print("符合要求了：", list(range(begin, min(len(end_list) - 1, begin + interval - 1))))
            tmp = {"video_name": filename,
                   # 我觉得很奇怪？为什么这个地方的截的是begin到begin+interval-1？ 为什么要减1呢？
                   "time": list(range(begin, min(len(end_list) - 1, begin + interval - 1))),
                   "length": l,
                   "is_full": is_full}

            if tmp not in results_60t120:
                results_60t120.append(tmp)

            begin = begin + interval - 1
            interval = 8

    # 两分钟到五分钟
    begin = 0
    interval = 16
    while begin < len(end_list) - 1:
        current_list = end_list[begin: begin + interval]
        # current_value_list = value_list[begin: begin + interval - 1]
        l = current_list[-1] - current_list[0]

        # value_score = sum(current_value_list) / len(current_value_list)

        if l <= 120:
            begin += 1
        elif l >= 300:
            interval -= 1
            if interval < 6:
                begin += 1
                interval = 16
        else:
            if list(range(begin, begin + interval - 1)) == list(range(0, len(end_list)-1)):
                    is_full = True
            else:
                    is_full = False
            # 符合条件
            tmp = {
                "video_name": filename,
                "time": list(range(begin, min(len(end_list) - 1, begin + interval - 1))),
                "length": l,
                "is_full": is_full
            }
            if tmp not in results_120t300:
                results_120t300.append(tmp)
            begin = begin + interval - 1
            interval = 16

    # 五分钟到十分钟
    begin = 0
    interval = len(end_list)
    while begin < len(end_list) - 1:
        current_list = end_list[begin: begin + interval]
        # current_value_list = value_list[begin: begin + interval - 1]
        l = current_list[-1] - current_list[0]

        # value_score = sum(current_value_list) / len(current_value_list)

        if l <= 300:
            begin += 1
        elif l >= 600:
            interval -= 1
            if interval < 6:
                begin += 1
                interval = len(end_list)
        else:
            if list(range(begin, begin + interval - 1)) == list(range(0, len(end_list)-1)):
                    is_full = True
            else:
                    is_full = False
            # 符合条件
            tmp = {"video_name": filename,
                   "time": list(range(begin, min(len(end_list) - 1, begin + interval - 1))),
                   "length": l,
                   "is_full": is_full }
            if tmp not in results_300t600:
                results_300t600.append(tmp)
            begin = begin + interval - 1
            begin -= interval // 5
            interval = len(end_list)


    begin = 0
    interval = len(end_list)
    while begin < len(end_list) - 1:
        current_list = end_list[begin: begin + interval]
        l = current_list[-1] - current_list[0]

        if l <= 600:
            begin += 1
        elif l >= 900:
            interval -= 1
            if interval < 6:
                begin += 1
                interval = len(end_list)
        else:
            if list(range(begin, begin + interval - 1)) == list(range(0, len(end_list)-1)):
                    is_full = True
            else:
                    is_full = False

            tmp = {"video_name": filename,
                "time": list(range(begin, min(len(end_list) - 1, begin + interval - 1))),
                "length": l,
                "is_full": is_full }
            if tmp not in results_600t900:
                results_600t900.append(tmp)
            begin = begin + interval - 1
            begin -= interval // 5
            interval = len(end_list)

    # 十五分钟到二十分钟
    begin = 0
    interval = len(end_list)
    while begin < len(end_list) - 1:
        current_list = end_list[begin: begin + interval]
        l = current_list[-1] - current_list[0]

        if l <= 900:
            begin += 1
        elif l >= 1200:
            interval -= 1
            if interval < 6:
                begin += 1
                interval = len(end_list)
        else:
            if list(range(begin, begin + interval - 1)) == list(range(0, len(end_list)-1)):
                    is_full = True
            else:
                    is_full = False

            tmp = {"video_name": filename,
                "time": list(range(begin, min(len(end_list) - 1, begin + interval - 1))),
                "length": l,
                "is_full": is_full}
            if tmp not in results_900t1200:
                results_900t1200.append(tmp)
            begin = begin + interval - 1
            begin -= interval // 5
            interval = len(end_list)

    # 二十分钟到二十五分钟
    begin = 0
    interval = len(end_list)
    while begin < len(end_list) - 1:
        current_list = end_list[begin: begin + interval]
        l = current_list[-1] - current_list[0]

        if l <= 1200:
            begin += 1
        elif l >= 1500:
            interval -= 1
            if interval < 6:
                begin += 1
                interval = len(end_list)
        else:
            if list(range(begin, begin + interval - 1)) == list(range(0, len(end_list)-1)):
                    is_full = True
            else:
                    is_full = False

            tmp = {"video_name": filename,
                "time": list(range(begin, min(len(end_list) - 1, begin + interval - 1))),
                "length": l,
                "is_full": is_full}
            if tmp not in results_1200t1500:
                results_1200t1500.append(tmp)
            begin = begin + interval - 1
            begin -= interval // 5
            interval = len(end_list)

    # 二十五分钟到三十分钟
    begin = 0
    interval = len(end_list)
    while begin < len(end_list) - 1:
        current_list = end_list[begin: begin + interval]
        l = current_list[-1] - current_list[0]
        is_full = True

        if l <= 1500:
            begin += 1
        elif l >= 1800:
            interval -= 1
            if interval < 6:
                begin += 1
                interval = len(end_list)
        else:
            if list(range(begin, begin + interval - 1)) == list(range(0, len(end_list)-1)):
                    is_full = True
            else:
                    is_full = False
                
            tmp = {"video_name": filename,
                "time": list(range(begin, min(len(end_list) - 1, begin + interval - 1))),
                "length": l,
                "is_full": is_full}
            if tmp not in results_1500t1800:
                results_1500t1800.append(tmp)
            begin = begin + interval - 1
            begin -= interval // 5
            interval = len(end_list)

    # 三十分钟以上
    if end_list[-1] >= 1800:
        begin = 0
        interval = 2
        while begin + interval < len(end_list) - 1:
            current_list = end_list[begin: begin + interval]
            l = current_list[-1] - current_list[0]
            is_full = True

            if l <= 1800:
                interval += 1
            else:
                if list(range(begin, begin + interval - 1)) == list(range(0, len(end_list)-1)):
                    is_full = True
                else:
                    is_full = False

                tmp = {"video_name": filename,
                    "time": list(range(begin, begin + interval - 1)),
                    "length": l,
                    "is_full": is_full}
                if tmp not in results_1800tu:
                    results_1800tu.append(tmp)
                begin = begin + interval - 1
                begin -= interval // 5
                interval = 2
        

# 完整视频的合并
    begin = 0
    l = end_list[-1] - end_list[0]
    tmp = {"video_name": filename,
            "time": list(range(begin, len(end_list)-1)),
            "length": l,
            "full": True}
    
    if tmp not in results_full:
        results_full.append(tmp)
    

    
    

print("60t120",len(results_60t120))
print("120t300",len(results_120t300))
print("300t600",len(results_300t600))
print("600t900",len(results_600t900))
print("900t1200",len(results_900t1200))
print("1200t1500",len(results_1200t1500))
print("1500t1800",len(results_1500t1800))
print("1800tu",len(results_1800tu))

json.dump(results_60t120, open(f"{args.save_path}/{args.save_name}_60t120.json", "w"))
json.dump(results_120t300, open(f"{args.save_path}/{args.save_name}_120t300.json", "w"))
json.dump(results_300t600, open(f"{args.save_path}/{args.save_name}_300t600.json", "w"))
json.dump(results_600t900, open(f"{args.save_path}/{args.save_name}_600t900.json", "w"))
json.dump(results_900t1200, open(f"{args.save_path}/{args.save_name}_900t1200.json", "w"))
json.dump(results_1200t1500, open(f"{args.save_path}/{args.save_name}_1200t1500.json", "w"))
json.dump(results_1500t1800, open(f"{args.save_path}/{args.save_name}_1500t1800.json", "w"))
json.dump(results_1800tu, open(f"{args.save_path}/{args.save_name}_1800tu.json", "w"))

json.dump(results_full, open(f"{args.save_path}/{args.save_name}_full.json", "w"))
